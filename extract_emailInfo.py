import json
import boto3
import email
#import sagemaker
from sms_spam_classifier_utilities import one_hot_encode, vectorize_sequences
#from sagemaker.mxnet.model import MXNetPredictor

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    session = boto3.Session()
    s3_session = session.client('s3')
    
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    data = s3_session.get_object(Bucket=bucket, Key=key)
    
    email_obj = email.message_from_bytes(data['Body'].read()) #get the whole email object
    from_email = email_obj.get('From') #get the from email address 
    body = email_obj.get_payload()[0].get_payload() #get the email body
    print("The email body: "+body)
    print("The email is from: "+from_email)
    print(type(from_email))
    
    endpoint_name = 'sms-spam-classifier-mxnet-2022-04-20-23-11-46-688'
    sageMaker = session.client('sagemaker-runtime')
    vocabulary_length = 9013
    input_mail = [body.strip()]
    print("The input_mail: "+str(input_mail)) 
    
    one_hot_test_messages = one_hot_encode(input_mail, vocabulary_length)
    print(one_hot_encode) #test purpose
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    print("The encoded_test_messages: "+ str(encoded_test_messages))
    data = json.dumps(encoded_test_messages.tolist())
    
    respond = sageMaker.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=data)
    
    res = json.loads(respond["Body"].read()) #read the result after using model 
    print("The respond: "+str(res))
    # get label and probability
    if res['predicted_label'][0][0] == 0:
        label = 'Not-A-spam'
    else:
        label = 'Spam'
    score = round(res['predicted_probability'][0][0], 4)
    score = score*100
    if label=='Not-A-spam':
        score = 100-score
    print(str(label) +" "+ str(score))
    
    message = "We received your email sent at " + str(email_obj.get('To')) + " with the subject " + str(email_obj.get('Subject')) + ".\n \
              Here is a 240 character sample of the email body:\n\n" + body[:240] + "\n \
              The email was categorized as " + str(label) + " with a " + str(score) + "% confidence."
    
    print("response_email: "+message)
    #from_email = 'Meng Zhou <mz3043@nyu.edu>'
    email_client = session.client('ses')
    response_email = email_client.send_email(
        Source='mz3043@mengzhou11.com',
        Destination={'ToAddresses': [str(from_email)]},
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': message,
                },
            },
            'Subject': {
                'Charset': 'UTF-8',
                'Data': 'Spam Analysis of Your Email',
            },
        }
    )

    print("Email sent!")
    return {}
    