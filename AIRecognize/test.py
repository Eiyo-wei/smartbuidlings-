from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'iic/cv_tinynas_object-detection_damoyolo_safety-helmet'
input_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_safetyhat.jpg'

safety_hat_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id, trust_remote_code=True)
result = safety_hat_detection(input_location)
print("result is : ", result)