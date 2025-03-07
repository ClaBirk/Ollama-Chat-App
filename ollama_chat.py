import sys
import json
import time
import re
import requests
from PyQt5 import QtWidgets, QtGui, QtCore

# -------------------------------------------------------
# Global Variables
# -------------------------------------------------------
conversation_history = []  # Each element is a complete turn (including prefix)
selected_model = "qwq-32b-ctx16k"
available_models = []
ollama_server = "http://192.168.3.1:11434/api"

# -------------------------------------------------------
# 1) Model Fetching
# -------------------------------------------------------
def fetch_models():
    global available_models, selected_model, ollama_server
    try:
        response = requests.get(f"{ollama_server}/tags", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            if "models" in data:
                # Preserve full model names with tags
                available_models = [m["model"] for m in data["models"]]  # Changed from "name" to "model"
            else:
                available_models = []
            if not available_models:
                available_models = [selected_model]
            if selected_model not in available_models:
                available_models.insert(0, selected_model)
            return True
        else:
            print(f"DEBUG: Server returned error {response.status_code}: {response.text}")
            available_models = [selected_model]
            return False
    except requests.exceptions.RequestException as e:
        print(f"DEBUG: Exception while fetching models: {e}")
        available_models = [selected_model]
        return False

# -------------------------------------------------------
# 2) Newline Collapsing
# -------------------------------------------------------
def collapse_newlines(text: str) -> str:
    """
    Replace 2+ consecutive newlines with a single newline
    to avoid huge vertical gaps.
    """
    return re.sub(r'\n{2,}', '\n', text)

# -------------------------------------------------------
# 3) Thinking Token Highlight
# -------------------------------------------------------
def highlight_thinking_tokens(text: str) -> str:
    """
    Replace <think> and </think> (and <thinking>, </thinking>) with
    a span so they display as literal text in bright yellow.
    """
    text = text.replace("<thinking>", '<span style="color: #FFD700;">&lt;thinking&gt;</span>')
    text = text.replace("</thinking>", '<span style="color: #FFD700;">&lt;/thinking&gt;</span>')
    text = text.replace("<think>", '<span style="color: #FFD700;">&lt;think&gt;</span>')
    text = text.replace("</think>", '<span style="color: #FFD700;">&lt;/think&gt;</span>')
    return text

# -------------------------------------------------------
# 4) Build HTML for Chat History
# -------------------------------------------------------
def build_html_chat_history():
    lines_html = []
    for line in conversation_history:
        line = collapse_newlines(line)
        if line.startswith("User:\n"):  # Modified here
            content = line[len("User:\n"):]
            content = highlight_thinking_tokens(content).replace("\n", "<br/>")
            line_html = f'<span style="color: orange;">User:</span><br/>{content}'
            lines_html.append(line_html)
        elif line.startswith("AI:\n"):
            content = line[len("AI:\n"):]
            content = highlight_thinking_tokens(content).replace("\n", "<br/>")
            line_html = f'<span style="color: orange;">AI:</span><br/>{content}'
            lines_html.append(line_html)
        else:
            line_html = highlight_thinking_tokens(line).replace("\n", "<br/>")
            lines_html.append(line_html)
    joined_html = "<br/>".join(lines_html)
    final_html = f"<div style='line-height:1.1; margin:0; padding:0;'>{joined_html}</div>"
    return final_html

# -------------------------------------------------------
# 5) Horizontal Separator
# -------------------------------------------------------
def create_separator():
    sep = QtWidgets.QFrame()
    sep.setFrameShape(QtWidgets.QFrame.HLine)
    sep.setFrameShadow(QtWidgets.QFrame.Sunken)
    sep.setLineWidth(1)
    sep.setStyleSheet("background-color: #262626;")
    return sep

# -------------------------------------------------------
# 6) Auto-resizing Text Edit
# -------------------------------------------------------
class AutoResizeTextEdit(QtWidgets.QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.document().contentsChanged.connect(self.adjust_height)
        self.max_height = 1000  # Will be set dynamically
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(60)  # Minimum height for good usability
        
    def adjust_height(self):
        # Calculate the height of the document
        doc_height = self.document().size().height()
        doc_margin = self.document().documentMargin()
        content_height = doc_height + 2 * doc_margin + 10  # Add a small buffer
        
        # Constrain to max_height
        new_height = min(content_height, self.max_height)
        
        # Ensure minimum height
        new_height = max(new_height, 60)
        
        # Apply the new height
        if new_height != self.height():
            self.setFixedHeight(new_height)
    
    def set_max_height(self, height):
        self.max_height = max(height, 60)  # Ensure minimum reasonable height
        self.adjust_height()

# -------------------------------------------------------
# 7) Worker Thread for Streaming Responses
# -------------------------------------------------------
class RequestWorker(QtCore.QObject):
    newChunk = QtCore.pyqtSignal(str)
    tokenCountUpdate = QtCore.pyqtSignal(int, int)  # used_tokens, max_tokens
    finished = QtCore.pyqtSignal()
    connectionError = QtCore.pyqtSignal(str)  # New signal for connection errors

    def __init__(self, prompt, current_history, max_context):
        super().__init__()
        self.prompt = prompt
        self.current_history = current_history.copy()
        self.max_context = max_context
        self.prompt_eval = 0
        self.eval_count = 0
        self.ai_response = ""
        # Rough estimate of tokens in the prompt (characters / 4)
        self.estimated_prompt_tokens = len(prompt) // 4
        self.accumulated_chunks = ""
        self.last_emit_time = time.time()
        
    def run(self):
        global selected_model, ollama_server
        self.ai_response = ""
        full_prompt = "\n".join(self.current_history)
        payload = {"model": selected_model, "prompt": full_prompt, "stream": True}

        # Initialize counters
        prompt_eval = 0
        eval_count = 0
        last_response_length = 0
        
        # Start with a blank token count - we'll only update at the end
        self.tokenCountUpdate.emit(0, self.max_context)
        print(f"DEBUG: Initial token count cleared, will update at completion")

        for attempt in range(3):  # Reduced attempts to 3
            try:
                with requests.post(f"{ollama_server}/generate", json=payload, stream=True, timeout=30) as response:
                    if response.status_code == 200:
                        for chunk in response.iter_lines():
                            if chunk:
                                # Handle SSE "data: " prefix
                                decoded_chunk = chunk.decode('utf-8').strip()
                                if decoded_chunk.startswith('data: '):
                                    decoded_chunk = decoded_chunk[len('data: '):]
                                
                                try:
                                    data = json.loads(decoded_chunk)
                                    
                                    # Get token counts if available (but don't emit updates yet)
                                    api_prompt_eval = data.get('prompt_eval_count', 0)
                                    api_eval_count = data.get('eval_count', 0)
                                    
                                    # Update our tracking if API provides values
                                    if api_prompt_eval > 0:
                                        prompt_eval = api_prompt_eval
                                    if api_eval_count > 0:
                                        eval_count = api_eval_count
                                    
                                    # Process response token
                                    if "response" in data:
                                        token_text = collapse_newlines(data["response"])
                                        self.ai_response += token_text
                                        
                                        # Accumulate chunks and emit less frequently for smoother UI
                                        self.accumulated_chunks += token_text
                                        current_time = time.time()
                                        
                                        # Emit chunks if enough time has passed or we have enough content
                                        if current_time - self.last_emit_time > 0.1 or len(self.accumulated_chunks) > 20:
                                            self.newChunk.emit(self.accumulated_chunks)
                                            self.accumulated_chunks = ""
                                            self.last_emit_time = current_time
                                        
                                        # If the API isn't providing token counts, estimate based on response length
                                        if api_eval_count == 0 and len(self.ai_response) > last_response_length:
                                            # Roughly estimate new tokens (4 chars ~= 1 token)
                                            new_chars = len(self.ai_response) - last_response_length
                                            new_tokens = max(1, new_chars // 4)
                                            eval_count += new_tokens
                                            last_response_length = len(self.ai_response)
                                    
                                    if data.get("done", False):
                                        # Emit any remaining accumulated chunks
                                        if self.accumulated_chunks:
                                            self.newChunk.emit(self.accumulated_chunks)
                                        
                                        # Only emit the final token count update when done
                                        final_tokens = prompt_eval + eval_count
                                        print(f"DEBUG: Final token count: {final_tokens}")
                                        self.tokenCountUpdate.emit(final_tokens, self.max_context)
                                        self.finished.emit()
                                        return
                                        
                                except json.JSONDecodeError:
                                    print(f"DEBUG: Failed to parse chunk: {decoded_chunk}")
                                    continue
                        self.finished.emit()
                        return
                    else:
                        # Handle HTTP errors
                        error_msg = f"Server error: {response.status_code}"
                        print(f"DEBUG: {error_msg}")
                        if attempt == 2:  # Last attempt
                            self.connectionError.emit(error_msg)
                            self.newChunk.emit(f"\n❌ {error_msg}")
                            self.finished.emit()
                            return
            except requests.exceptions.ConnectionError:
                error_msg = "Connection error"
                print(f"DEBUG: {error_msg}")
                time.sleep(1)
                if attempt == 2:  # Last attempt
                    self.connectionError.emit(error_msg)
                    self.newChunk.emit(f"\n❌ Failed to connect after 3 attempts.")
                    self.finished.emit()
                    return
            except requests.exceptions.Timeout:
                error_msg = "Request timeout"
                print(f"DEBUG: {error_msg}")
                self.connectionError.emit(error_msg)
                self.newChunk.emit("\n⏳ Server timeout! Try a shorter request.")
                self.finished.emit()
                return
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                print(f"DEBUG: {error_msg}")
                self.connectionError.emit(error_msg)
                self.newChunk.emit(f"\n❌ Error: {error_msg}")
                self.finished.emit()
                return

# -------------------------------------------------------
# 8) Main Chat Window
# -------------------------------------------------------
class ChatWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ollama Chat")
        self.resize(900, 650)
        self.current_ai_text = ""
        self.current_ai_index = -1
        self.thread = None
        self.worker = None
        self.last_token_count = 0  # Store the last known token count
        self.api_connected = False  # Track API connection status
        self.current_model_context = 4096  # Default context size
        
        # Set up the UI
        self.setupUI()
        
        # Setup API connection checker timer
        self.api_check_timer = QtCore.QTimer(self)
        self.api_check_timer.timeout.connect(self.check_api_connection)
        self.api_check_timer.start(3000)  # Check every 3 seconds
        
        # Initial connection check
        QtCore.QTimer.singleShot(100, self.check_api_connection)
        
    def closeEvent(self, event):
        # Stop the timer before closing
        if hasattr(self, 'api_check_timer'):
            self.api_check_timer.stop()
        
        # Clean up any running thread
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(1000)  # Wait up to 1 second
        
        # Accept the close event
        event.accept()
        
    def fetch_model_context(self, model_name: str):
        """Fetch the model's context window size using comprehensive checks"""
        try:
            response = requests.post(
                f"{ollama_server}/show",
                json={"name": model_name},
                timeout=2.0
            )
            if response.status_code == 200:
                data = response.json()
                max_context = None

                # 1. Check parameters string for num_ctx
                if "parameters" in data and isinstance(data["parameters"], str):
                    params = data["parameters"].lower()
                    if "num_ctx" in params:
                        for line in params.split('\n'):
                            if "num_ctx" in line:
                                try:
                                    max_context = int(line.split()[-1])
                                    break
                                except (ValueError, IndexError):
                                    pass

                # 2. Check model_info for architecture-specific context
                if not max_context:
                    arch = data.get("details", {}).get("family", "").lower()
                    context_key = f"{arch}.context_length" if arch else "context_length"
                    max_context = data.get("model_info", {}).get(context_key)

                # 3. Check model_info for general context_length
                if not max_context:
                    max_context = data.get("model_info", {}).get("context_length")

                # Use the context or fall back to default
                if max_context is not None:
                    self.current_model_context = max_context
                    print(f"Context size updated: {max_context}")
                    return True
            
            print(f"API Error or couldn't determine context size, using default")
            return False
            
        except Exception as e:
            print(f"Context fetch error: {str(e)}")
            return False

    def change_model(self, new_model):
        global selected_model
        selected_model = new_model
        print(f"DEBUG: Selected model changed to: {new_model}")
        success = self.fetch_model_context(new_model)
        self.token_count_label.setText(f"Tokens: {self.last_token_count} / {self.current_model_context}")
    
    def setupUI(self):
        main_layout = QtWidgets.QVBoxLayout()

        bold_label_font = QtGui.QFont()
        bold_label_font.setPointSize(11)
        bold_label_font.setBold(True)
        
        # --- Server Address ---
        server_label = QtWidgets.QLabel("Server Address:")
        server_label.setAlignment(QtCore.Qt.AlignLeft)
        server_label.setFont(bold_label_font)
        main_layout.addWidget(server_label)

        row_server_layout = QtWidgets.QHBoxLayout()
        self.server_input = QtWidgets.QLineEdit(ollama_server)
        self.server_input.setFixedWidth(200)
        
        # Connection status indicator (circular)
        self.connection_indicator = QtWidgets.QLabel()
        self.connection_indicator.setFixedSize(16, 16)
        self.connection_indicator.setStyleSheet("""
            background-color: #000000;
            border-radius: 8px;
            margin: 2px;
        """)
        # Create a tooltip for the indicator
        self.connection_indicator.setToolTip("Disconnected")
        
        # Connect/Disconnect button
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)
        
        row_server_layout.addWidget(self.server_input)
        row_server_layout.addWidget(self.connection_indicator)
        row_server_layout.addWidget(self.connect_button)
        row_server_layout.addStretch(1)
        main_layout.addLayout(row_server_layout)
        main_layout.addWidget(create_separator())

        # --- Select Model ---
        model_label = QtWidgets.QLabel("Select Model:")
        model_label.setAlignment(QtCore.Qt.AlignLeft)
        model_label.setFont(bold_label_font)
        main_layout.addWidget(model_label)

        row_model_layout = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setFixedWidth(450)
        self.model_combo.currentTextChanged.connect(self.change_model)
        row_model_layout.addWidget(self.model_combo)
        
        self.token_count_label = QtWidgets.QLabel("Tokens: 0 / 0")
        self.token_count_label.setFont(bold_label_font)
        row_model_layout.addWidget(self.token_count_label)
        
        row_model_layout.addStretch(1)
        main_layout.addLayout(row_model_layout)
        main_layout.addWidget(create_separator())

        # --- Chat History ---
        chat_label = QtWidgets.QLabel("Chat History:")
        chat_label.setAlignment(QtCore.Qt.AlignLeft)
        chat_label.setFont(bold_label_font)
        main_layout.addWidget(chat_label)

        # Create a container widget for the chat and prompt areas
        self.chat_prompt_container = QtWidgets.QWidget()
        chat_prompt_layout = QtWidgets.QVBoxLayout(self.chat_prompt_container)
        chat_prompt_layout.setContentsMargins(0, 0, 0, 0)
        
        self.chat_history_widget = QtWidgets.QTextEdit()
        self.chat_history_widget.setObjectName("ChatHistory")
        self.chat_history_widget.setReadOnly(True)
        self.chat_history_widget.setWordWrapMode(QtGui.QTextOption.WordWrap)
        
        # --- Prompt ---
        prompt_label = QtWidgets.QLabel("Enter your prompt:")
        prompt_label.setAlignment(QtCore.Qt.AlignLeft)
        prompt_label.setFont(bold_label_font)
        
        # Auto-resizing prompt input
        self.prompt_input = AutoResizeTextEdit()
        self.prompt_input.setObjectName("PromptInput")
        self.prompt_input.setWordWrapMode(QtGui.QTextOption.WordWrap)
        
        # Add widgets to the chat_prompt_layout
        chat_prompt_layout.addWidget(self.chat_history_widget, 1)  # Chat takes remaining space
        chat_prompt_layout.addWidget(create_separator())
        chat_prompt_layout.addWidget(prompt_label)
        chat_prompt_layout.addWidget(self.prompt_input, 0)  # Prompt has no stretch factor (sized by content)
        
        # Add the container to the main layout
        main_layout.addWidget(self.chat_prompt_container, 1)
        main_layout.addWidget(create_separator())

        # Status indicator label
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #FF9500; font-weight: bold;")
        main_layout.addWidget(self.status_label)
        
        buttons_layout = QtWidgets.QHBoxLayout()
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_chat)
        self.exit_button = QtWidgets.QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        buttons_layout.addWidget(self.send_button)
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.exit_button)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

        self.server_input.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.model_combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        
        # Set initial prompt height
        self.updatePromptHeight()

    def showEvent(self, event):
        super().showEvent(event)
        # Set button height
        button_height = self.connect_button.sizeHint().height()
        self.server_input.setFixedHeight(button_height)
        self.model_combo.setFixedHeight(button_height)
        
        # Initial resize of prompt input
        QtCore.QTimer.singleShot(100, self.updatePromptHeight)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # When window is resized, update the prompt input max height
        self.updatePromptHeight()
    
    def updatePromptHeight(self):
        """Update the prompt input height based on the golden ratio"""
        if hasattr(self, 'chat_prompt_container') and hasattr(self.prompt_input, 'set_max_height'):
            container_height = self.chat_prompt_container.height()
            max_prompt_height = int(container_height * 0.382)  # Golden ratio proportion
            self.prompt_input.set_max_height(max_prompt_height)
            # Also trigger adjustment to current content
            self.prompt_input.adjust_height()

    def check_api_connection(self):
        """Check if the Ollama API server is accessible and update UI accordingly"""
        global ollama_server
        
        # Don't perform check if a generation is in progress
        if not self.send_button.isEnabled() and self.thread is not None and self.thread.isRunning():
            return
            
        try:
            # Use a short timeout to avoid blocking the UI for too long
            response = requests.get(f"{ollama_server}/version", timeout=1.0)
            if response.status_code == 200:
                if not self.api_connected:
                    self.api_connected = True
                    self.update_connect_button()
                    print("DEBUG: API connection established")
                    # Update models since we're now connected
                    if fetch_models():
                        self.update_model_combo()
                        self.fetch_model_context(selected_model)
            else:
                if self.api_connected:
                    self.api_connected = False
                    self.update_connect_button()
                    print(f"DEBUG: API connection lost (status {response.status_code})")
        except requests.exceptions.RequestException:
            if self.api_connected:
                self.api_connected = False
                self.update_connect_button()
                print("DEBUG: API connection lost (connection error)")
    
    def update_connect_button(self):
        """Update the connect button appearance based on connection status"""
        if self.api_connected:
            # Connected - green indicator and Disconnect button
            self.connection_indicator.setStyleSheet("""
                background-color: #2A5E2A;
                border: 1px solid #3E8E3E;
                border-radius: 8px;
                margin: 2px;
            """)
            self.connection_indicator.setToolTip("Connected to Ollama API")
            self.connect_button.setText("Disconnect")
            # Update UI elements when connected
            self.send_button.setEnabled(True)
            self.reset_button.setEnabled(True)
        else:
            # Not connected - black indicator and Connect button
            self.connection_indicator.setStyleSheet("""
                background-color: #000000;
                border: 1px solid #404040;
                border-radius: 8px;
                margin: 2px;
            """)
            self.connection_indicator.setToolTip("Disconnected")
            self.connect_button.setText("Connect")
            # Disable send button when disconnected
            self.send_button.setEnabled(False)
    
    def toggle_connection(self):
        """Toggle between connecting and disconnecting from the server"""
        if self.api_connected:
            # Currently connected, so disconnect
            self.api_connected = False
            self.update_connect_button()
            # Disable message sending when disconnected
            self.send_button.setEnabled(False)
            print("DEBUG: Manually disconnected from server")
        else:
            # Currently disconnected, so connect
            self.connect_server()
            
    def connect_server(self):
        """Connect to the specified Ollama server"""
        global ollama_server
        new_server = self.server_input.text().strip()
        if new_server:
            ollama_server = new_server
            # Reset connection status and start checking
            self.api_connected = False
            self.update_connect_button()
            self.check_api_connection()  # Immediate check after change
            
    def update_model_combo(self):
        global available_models
        current_text = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItems(available_models)
        
        # Try to restore the previous selection
        index = self.model_combo.findText(current_text)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

    def update_token_count(self, used: int, maximum: int):
        # Only update the token count display with the final count
        if used > 0:  # Only show non-zero token counts (final result)
            print(f"DEBUG: Displaying final token count: {used} / {maximum}")
            self.last_token_count = used  # Store the token count for future reference
            self.token_count_label.setText(f"Tokens: {used} / {maximum}")
            self.token_count_label.repaint()
            # Process events to ensure the UI updates immediately
            QtWidgets.QApplication.processEvents()

    def send_message(self):
        global conversation_history
        
        # If not connected, show error and return
        if not self.api_connected:
            self.status_label.setText("❌ Not connected to API server!")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            return
            
        prompt = self.prompt_input.toPlainText().strip()
        if prompt:
            # Show the previous token count with a "+" indicator
            if self.last_token_count > 0:
                self.token_count_label.setText(f"Tokens: {self.last_token_count}+ / {self.current_model_context}")
            else:
                self.token_count_label.setText(f"Tokens: 0 / {self.current_model_context}")
            
            # Append user message and AI placeholder
            conversation_history.append(f"User:\n{prompt}")
            conversation_history.append("AI:\n")
            self.current_ai_index = len(conversation_history) - 1
            
            # Immediately update UI to show user's message
            self.update_chat_history()
            
            self.current_ai_text = ""
            self.prompt_input.clear()
            
            # Set focus to the chat history widget to prevent marking the API address
            self.chat_history_widget.setFocus()
            
            # Disable all buttons while generating response
            self.send_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.exit_button.setEnabled(False)
            
            # Update status indicator
            self.status_label.setText("⟳ Generating response... Please wait.")
            
            self.start_worker(prompt, conversation_history[:])
        
    def start_worker(self, prompt, current_history):
        # Check if previous thread is still running and properly clean up
        if self.thread is not None:
            if self.thread.isRunning():
                print("DEBUG: Stopping running thread.")
                self.thread.quit()
                self.thread.wait(1000)  # Wait up to 1 second
                print("DEBUG: Stopped and cleaned up the thread.")
            self.thread = None
            self.worker = None

        # Create new thread and worker
        print("DEBUG: Creating a new thread and worker.")
        self.thread = QtCore.QThread()
        self.worker = RequestWorker(prompt, current_history, self.current_model_context)
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.worker.tokenCountUpdate.connect(self.update_token_count)
        self.worker.newChunk.connect(self.handle_new_chunk)
        self.worker.finished.connect(self.handle_finished)
        self.worker.connectionError.connect(self.handle_connection_error)
        self.thread.started.connect(self.worker.run)

        # Cleanup connections
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Reset references when objects are destroyed
        self.thread.destroyed.connect(lambda: setattr(self, 'thread', None))
        self.worker.destroyed.connect(lambda: setattr(self, 'worker', None))

        # Start the thread
        self.thread.start()
        print("DEBUG: Thread started.")

    def handle_connection_error(self, error_msg):
        # Update connection status on error
        self.api_connected = False
        self.update_connect_button()
        # Trigger immediate API check
        QtCore.QTimer.singleShot(500, self.check_api_connection)

    def handle_new_chunk(self, chunk):
        # Tokenize the current text and the incoming chunk
        current_tokens = self.current_ai_text.split()
        new_tokens = chunk.split()

        # Check if the new tokens are already at the end of current_tokens
        if current_tokens and new_tokens and len(current_tokens) >= len(new_tokens) and current_tokens[-len(new_tokens):] == new_tokens:
            print("DEBUG chunk received (DUPLICATE IGNORED):", repr(chunk))
            return

        # Store current scroll position before updating
        scrollbar = self.chat_history_widget.verticalScrollBar()
        current_scroll = scrollbar.value()
        
        # Update the conversation history
        self.current_ai_text += chunk
        conversation_history[self.current_ai_index] = "AI:\n" + self.current_ai_text
        
        # Update display with scroll position preservation
        self.update_chat_history(preserve_scroll=current_scroll)
        
        # Process events to keep the UI responsive
        QtWidgets.QApplication.processEvents()

    def handle_finished(self):
        # Re-enable all buttons when response is complete
        self.send_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.exit_button.setEnabled(True)
        
        # Clear status indicator
        self.status_label.setText("")

    def reset_chat(self):
        global conversation_history
        conversation_history = []
        self.chat_history_widget.clear()
        self.last_token_count = 0  # Reset the token count on chat reset
        self.token_count_label.setText(f"Tokens: 0 / {self.current_model_context}")

    def update_chat_history(self, preserve_scroll=None):
        # Generate HTML content
        html_content = build_html_chat_history()
        
        # Update the chat history text
        self.chat_history_widget.setHtml(html_content)
        
        # Restore scroll position if specified
        if preserve_scroll is not None:
            scrollbar = self.chat_history_widget.verticalScrollBar()
            scrollbar.setValue(preserve_scroll)
            
        # Process events to keep UI responsive
        QtWidgets.QApplication.processEvents()

# -------------------------------------------------------
# 9) Dark Mode Styling
# -------------------------------------------------------
def apply_dark_mode(app):
    QtWidgets.QApplication.setStyle("Fusion")

    dark_palette = QtGui.QPalette()
    
    base_window_color = QtGui.QColor("#2f2f2f")
    chat_bg_color     = QtGui.QColor("#2a2a2a")
    alt_base_color    = QtGui.QColor("#3b3b3b")
    text_color        = QtGui.QColor("#ffffff")
    button_color      = QtGui.QColor("#3e3e3e")
    highlight_color   = QtGui.QColor("#537BA2")
    border_color      = QtGui.QColor("#4f4f4f")

    dark_palette.setColor(QtGui.QPalette.Window, base_window_color)
    dark_palette.setColor(QtGui.QPalette.WindowText, text_color)
    dark_palette.setColor(QtGui.QPalette.Base, alt_base_color)
    dark_palette.setColor(QtGui.QPalette.AlternateBase, base_window_color)
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, text_color)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, text_color)
    dark_palette.setColor(QtGui.QPalette.Text, text_color)
    dark_palette.setColor(QtGui.QPalette.Button, button_color)
    dark_palette.setColor(QtGui.QPalette.ButtonText, text_color)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Link, highlight_color)
    dark_palette.setColor(QtGui.QPalette.Highlight, highlight_color)
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)

    app.setPalette(dark_palette)

    app.setStyleSheet(f"""
        QWidget {{
            font-size: 10pt;
            color: {text_color.name()};
        }}
        QToolTip {{
            color: #ffffff;
            background-color: {highlight_color.name()};
            border: 1px solid {text_color.name()};
        }}
        QPushButton {{
            border: 1px solid {border_color.name()};
            background-color: {button_color.name()};
            padding: 6px;
        }}
        QPushButton:hover {{
            background-color: #4a4a4a;
        }}
        QPushButton:pressed {{
            background-color: #5a5a5a;
        }}
        QPushButton:disabled {{
            background-color: #282828;
            color: #606060;
            border: 1px solid #404040;
        }}
        QLineEdit, QComboBox {{
            background-color: {alt_base_color.name()};
            border: 1px solid {border_color.name()};
            color: {text_color.name()};
        }}
        QComboBox::drop-down {{
            border-left: 1px solid {border_color.name()};
        }}
        QTextEdit#ChatHistory {{
            background-color: #2a2a2a;
            border: 1px solid {border_color.name()};
            color: {text_color.name()};
        }}
        QTextEdit#PromptInput {{
            background-color: {alt_base_color.name()};
            border: 1px solid {border_color.name()};
            color: {text_color.name()};
        }}
        QScrollBar:vertical {{
            background-color: {alt_base_color.name()};
            width: 12px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background-color: #555555;
            min-height: 20px;
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            background: none;
            border: none;
            height: 0px;
        }}
    """)

# -------------------------------------------------------
# 10) Main Entry
# -------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_mode(app)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())