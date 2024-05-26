
# Finance WebApp

## Overview

Finance WebApp is a comprehensive tool designed to help users visualize financial data, extract financial filings, and interact with financial documents through a chatbot interface. This application leverages various APIs including Alpha Vantage, EDGAR, and OpenAI to provide users with real-time stock data, news sentiment analysis, and detailed financial filings.

## Features

- **Data Visualization**: Fetch and visualize financial data including technical indicators, commodities, economic indicators, and forex data.
- **Market News Sentiment**: Retrieve and display sentiment analysis on market news for selected tickers.
- **Real-Time Data Platform**: View and analyze real-time stock data with automatic updates.
- **Filing Extractor**: Extract specific sections from financial filings such as 10-K, 10-Q, and 8-K.
- **Filing ChatBot**: Ask questions about company filings and receive detailed responses powered by OpenAI.

## Repository Structure

\`\`\`plaintext
.
├── .chroma
├── assets
│   └── image.webp
├── data
├── filings
├── lightning_logs/version_0
├── notebooks
├── src
│   ├── app
│   │   ├── data
│   │   │   ├── __pycache__
│   │   │   ├── app.py
│   │   │   ├── fetch_data.py
│   │   │   ├── langchain_utils.py
│   │   │   ├── physical_currency_list.csv
│   │   │   ├── preprocessor.py
│   │   │   └── utils.py
│   ├── models
│   │   └── train.py
├── Dockerfile
├── environment.yaml
└── requirements.txt
\`\`\`

## Installation

1. **Clone the repository**
   \`\`\`bash
   git clone <repository-url>
   cd <repository-directory>
   \`\`\`

2. **Set up the environment**
   - Using `conda`:
     \`\`\`bash
     conda env create -f environment.yaml
     conda activate <env-name>
     \`\`\`
   - Using `pip`:
     \`\`\`bash
     pip install -r requirements.txt
     \`\`\`

3. **Run the application**
   \`\`\`bash
   streamlit run src/app/data/app.py
   \`\`\`

## Usage

### Navigation

- **Alpha Vantage - Data Visualization**: Input your Alpha Vantage API key and select the type of data you want to visualize.
- **Alpha Vantage - Market News Sentiment**: Enter the ticker symbols and fetch news sentiment analysis.
- **Real-Time Platform**: Provide the Alpha Vantage API key and view real-time stock data.
- **Filing Extractor**: Input the EDGAR API Identity and extract specific sections from filings.
- **Filing ChatBot**: Ask questions about company filings using OpenAI API and EDGAR API Identity.

### Example API Keys
- Alpha Vantage: Obtain from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).
- EDGAR: Register and obtain identity from [SEC EDGAR](https://www.sec.gov/edgar/sec-api-documentation).
- OpenAI: Obtain API key from [OpenAI](https://platform.openai.com/account/api-keys).

## Things to Fix or Add

### Fixes
1. **Ensure Proper File Paths**:
   - Correct the file paths for accessing CSV files and assets in the `app.py`.

2. **Error Handling**:
   - Add more robust error handling when fetching data from APIs to manage API limits and invalid inputs.
   - Improve error messages to guide users for troubleshooting.

3. **Documentation for Functions**:
   - Add docstrings for all functions to improve readability and maintainability.

4. **API Key Management**:
   - Securely manage API keys, possibly using environment variables or a configuration file.

### Additions
1. **Unit Tests**:
   - Add unit tests for the data extraction and processing functions to ensure reliability.

2. **Improved UI**:
   - Enhance the Streamlit interface with better layout and styling for a more user-friendly experience.

3. **Caching**:
   - Implement caching for API responses to improve performance and reduce the number of API calls.

4. **Expanded Data Sources**:
   - Include additional data sources or APIs to provide a broader range of financial data.

5. **Interactive Charts**:
   - Add more interactive elements to charts, such as tooltips and filters.

## Contribution

Feel free to contribute by opening issues, submitting pull requests, or providing feedback.

## License

This project is licensed under the MIT License.
