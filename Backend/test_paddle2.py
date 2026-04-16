import numpy as np
import traceback

def test_paddle():
    try:
        from paddleocr import PaddleOCR
        print("Importing OCR with enable_mkldnn=False, use_angle_cls=True")
        ocr = PaddleOCR(lang='en', enable_mkldnn=False, use_angle_cls=True)
        res = ocr.ocr(np.zeros((100, 100, 3), dtype=np.uint8))
        print("Success without MKLDNN but WITH Angle CLS")
        return True
    except Exception as e:
        print("Failed:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_paddle()
