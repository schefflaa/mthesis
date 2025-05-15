class InfoMessage:
    def __init__(self, statusCode: int, url: str, message: str, img: bytes = None):
        self.statusCode = statusCode
        self.url = url
        self.message = message

    def to_dict(self) -> dict:
        return {
            "statusCode": self.statusCode,
            "url": self.url,
            "message": self.message
        }
    
    def __str__(self) -> str:
        return f"statusCode: {self.statusCode}, url: {self.url}, message: {self.message}"
    