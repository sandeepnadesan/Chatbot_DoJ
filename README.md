# Chatbot_DoJ

**Chatbot_DoJ** is a chatbot designed to assist users by answering questions and providing information. The project uses natural language processing (NLP) techniques to interpret user queries and respond in a conversational manner.

## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Features
- **Conversational AI:** Interacts with users and responds to queries.
- **Natural Language Understanding:** Processes user inputs for better context-based responses.
- **Extensible:** Add custom functionalities and responses as per requirements.

## Technology Stack
- **Backend:** Python
- **Framework:** Flask (for API integration)
- **NLP Library:** NLTK or similar
- **Frontend:** HTML, JavaScript
- **Database:** MongoDB (optional, for logging conversations or custom user data)
- **Other Tools:** Docker, Git

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Flask
- MongoDB (if using for storage)
- Git

### Backend Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/sandeepnadesan/Chatbot_DoJ.git
    cd Chatbot_DoJ
    ```

2. Set up a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. Start the Flask server:
    ```bash
    python app.py
    ```

The Flask server should now be running at `https://chatbot-doj.streamlit.app/`.

## Usage
Once the server is running, users can interact with the chatbot through a web-based interface. The chatbot will respond to user queries and provide the relevant information.

If using MongoDB, the chatbot can log conversations or retrieve specific user data from the database.


