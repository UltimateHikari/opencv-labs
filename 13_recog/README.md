# Motivations

First, i was thinking about training CNN with keras for doing OCR, but got too lazy for searching a dataset for it.  

So, for image search, instead of thresholding, blackhat masks and erode+dilate (basically, stuff already done in 10_segmentation) here will be cascade classifier with opencv profile for russian plates  

And instead of segmentation of result and & feeding each single symbol to CNN - tesseract ocr.
