
1. Go to http://qiicr.org/dcmqi/#/seg to create a dicom seg template similar to dmc_seg_template.json
2. Everything in dmc_seg_template.json that is set to "null" currently will be overwritten with information from the original dicom image
3. Use that template when creating your SegmentationResult() objects