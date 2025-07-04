from flask import Flask
from api.optimize import app  # Import Flask app from optimize.py

import sys

# manage.py
def run_app():
    if len(sys.argv) > 1 and sys.argv[1] == 'production':
        app.run(host='0.0.0.0', port=80)  # Run on port 80 in production mode
    else:
        app.run(debug=True)  # Default: run in development mode

if __name__ == '__main__':
    run_app()
