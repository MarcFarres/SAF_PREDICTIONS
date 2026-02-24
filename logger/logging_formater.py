import logging

class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[37m",   # White/Grey
        logging.INFO: "\033[36m",    # Cyan
        logging.WARNING: "\033[33m", # Yellow
        logging.ERROR: "\033[31m",   # Red
        logging.CRITICAL: "\033[41m" # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"