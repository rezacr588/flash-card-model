# Flashcard Response Prediction Model

This project aims to predict user responses to flashcards based on various features like the time taken to respond, the difficulty of the flashcard, and the timestamp of the response.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

The model is built using the RandomForestClassifier from the Scikit-learn library. The dataset used is synthetic, generated based on certain assumptions about user behavior.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rezacr588/flash-card-model
   ```

2. Navigate to the project directory:
   ```bash
   cd flash-project
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:
   ```bash
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. To generate synthetic data:
   ```bash
   python data/generate_synthetic.py
   ```

2. To train and evaluate the model:
   ```bash
   python main.py
   ```

## Testing

To run the tests:
```bash
python -m unittest test_model.py
```

## Contributing

1. Fork the project.
2. Create your feature branch: `git checkout -b feature/YourFeatureName`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/YourFeatureName`
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.