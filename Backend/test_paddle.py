import numpy as np
import traceback

def test_paddle():
    try:
        from paddleocr import PaddleOCR
        print("Importing OCR with enable_mkldnn=False, use_angle_cls=False")
        ocr = PaddleOCR(lang='en', enable_mkldnn=False, use_angle_cls=False)
        res = ocr.ocr(np.zeros((100, 100, 3), dtype=np.uint8))
        print("Success without MKLDNN and without Angle CLS")
        return True
    except Exception as e:
        print("Failed without MKLDNN:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if not test_paddle():
        try:
            from paddleocr import PaddleOCR
            print("Importing OCR with use_onnx=True")
            ocr = PaddleOCR(lang='en', use_onnx=True)
            res = ocr.ocr(np.zeros((100, 100, 3), dtype=np.uint8))
            print("Success with ONNX")
        except Exception as e:
            traceback.print_exc()
