"""
WSGI entry point for production deployment
"""
import os
from web import app

# Ensure we're using CPU-only mode in production
if __name__ != '__main__':
    cpu_only = os.getenv('CPU_ONLY', '1') == '1'

# WSGI application
application = app

if __name__ == '__main__':
    # For local development
    application.run(host='0.0.0.0', port=5000, debug=False)