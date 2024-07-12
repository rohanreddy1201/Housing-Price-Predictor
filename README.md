# Real-Time Housing Price Predictor

## Introduction

The Real-Time Housing Price Predictor is a web application that predicts the housing prices using the California Housing dataset. This project leverages machine learning techniques to analyze housing data and provides a user-friendly interface for real-time predictions and visualizations.

## Need for Usage

With the real estate market constantly changing, investors, realtors, and buyers seek tools that can help them make informed decisions. The Real-Time Housing Price Predictor aims to:

- Provide real-time housing price predictions.
- Visualize recent housing data metrics and trends.
- Enhance decision-making processes for investors, realtors, and buyers.

## Tech Stack Used

- **Python**: Core programming language for data processing and machine learning.
- **Pandas**: Data manipulation and analysis.
- **Scikit-Learn**: Machine learning library for model training and evaluation.
- **Streamlit**: Web framework for building the user interface.
- **Matplotlib**: Visualization library for plotting housing data metrics.
- **Seaborn**: Data visualization library based on Matplotlib.
- **Requests**: For making HTTP requests (optional for real-time API).

## Features

- **Real-Time Predictions**: Predicts housing prices using real-time data.
- **User-Friendly Interface**: Streamlit-based web application for easy interaction.
- **Visualizations**: Displays housing data metrics and trends.
- **Sidebar Input**: Allows users to specify input parameters for predictions.

## Future Enhancements

- **Expanded Model**: Incorporate more sophisticated machine learning models for better accuracy.
- **Additional Metrics**: Include more real estate metrics and indicators for comprehensive analysis.
- **Geographical Analysis**: Provide insights and trends based on geographical data.
- **Notifications**: Alert users about significant changes in housing prices.

## How to Use

### Prerequisites

- Python 3.x installed on your system.
- A GitHub account.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/real-time-housing-price-predictor.git
   cd real-time-housing-price-predictor
   ```
2. **Install the Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application:

1. **Train the Model Using built-in California Housing Data in sklearn library:**
   ```bash
   python housing_predictor.py
   ```

2. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

3. **Access the App in Your Web Browser:**
   Open your web browser and go to 'http://localhost:8501'


## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to improve the project.

## Contact
For any questions or feedback, please contact [ackrohan@gmail.com].
