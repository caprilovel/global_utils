import sys,os,time
from datetime import datetime 
import pytz
from email.mime.multipart import MIMEMultipart
import smtplib
from email.mime.text import MIMEText
# 负责构造图片
from email.mime.image import MIMEImage
# 负责将多个对象集合起来
from email.mime.multipart import MIMEMultipart
from email.header import Header


def print_args(func):
    """wrapper for print args and kwargs

    Args:
        func (_type_): function
        
    Examples:
        >>> @print_args
        >>> def test(a,b,c):
        >>>    pass
        >>> test(1,2,3)
        -------------test KWArgs---------------
        a: 1
        b: 2
        c: 3
        -------------test Args---------------
    """
    def wrapper(*args, **kwargs):
        print("-------------{} KWArgs---------------".format(func.__name__))
        for k,v in kwargs.items():
            print("{}: {}".format(k,v))
        print("-------------{} Args---------------".format(func.__name__))
        return func(*args, **kwargs)
    return wrapper

def torch_log_decorator(func):
    def wrapper(*args, **kwargs):
        from torch.utils.tensorboard import SummaryWriter
        # 创建SummaryWriter对象，指定日志保存路径
        writer = SummaryWriter(log_dir="./logs")
        result = func(*args, **kwargs)
        # 在训练过程中记录日志
        writer.add_scalar("loss", result["loss"], global_step=result["step"])
        writer.add_scalar("accuracy", result["accuracy"], global_step=result["step"])
        writer.close()
        return result
    return wrapper

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_time_str(style='Nonetype'):
    t = time.localtime()
    if style is 'Nonetype':
        return ("{}{}{}{}{}{}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
    elif style is 'underline':
        return ("{}_{}_{}_{}_{}_{}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
    
def timestamp():
    time_format = '%Y%m%d%H%M%S'
    timer = datetime.now(pytz.timezone('Asia/Shanghai')).strftime(time_format)
    return timer

class Logger(object):
    def __init__(self, logFile='./Default.log'):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')
        self.close_flag = False
    
    def write(self, message):
        self.terminal.write(message)
        if self.close_flag is False:
            self.log.write(message)
    
    def flush(self):
        pass
    
    def close(self):
        self.log.close()
        self.close_flag = True


class EmailSender:
    def __init__(self, title="Message"):
        self.mail_host = "smtp.163.com"
        self.mail_sender = "caprilovel_3@163.com"
        self.mail_license = "STSGBJYKXKCONQGP"
        self.mail_receivers = ["1797257299@qq.com"]
        self.mm = MIMEMultipart('related')
        # 邮件主题
        subject_content = title
        # 设置发送者,注意严格遵守格式,里面邮箱为发件人邮箱
        self.mm["From"] = "capri_sender<caprilovel_3@163.com>"
        # 设置接受者,注意严格遵守格式,里面邮箱为接受者邮箱
        self.mm["To"] = "capri_receiver<1797257299@qq.com>"
        # 设置邮件主题
        self.mm["Subject"] = Header(subject_content,'utf-8')
        
    def get_text(self, text):
        # 邮件正文内容
        body_content = text
        # 构造文本,参数1：正文内容，参数2：文本格式，参数3：编码方式
        message_text = MIMEText(body_content,"plain","utf-8")
        # 向MIMEMultipart对象中添加文本对象
        self.mm.attach(message_text)
    
    def get_picture(self, image_data):
        # 二进制读取图片
        # image_data = open('a.jpg','rb')
        # 设置读取获取的二进制数据
        message_image = MIMEImage(image_data.read())
        # 关闭刚才打开的文件
        image_data.close()
        # 添加图片文件到邮件信息当中去
        self.mm.attach(message_image)
        
    def get_attachment(self, attachment_path):
        # 构造附件
        atta = MIMEText(open(attachment_path, 'rb').read(), 'base64', 'utf-8')
        # 设置附件信息
        atta["Content-Disposition"] = 'attachment; filename="file.log"'
        # 添加附件到邮件信息当中去
        self.mm.attach(atta)
        
    def send(self):
        try:
            # 创建SMTP对象
            stp = smtplib.SMTP()
            # stp = smtplib.SMTP_SSL(self.mail_host)
            # 设置发件人邮箱的域名和端口，端口地址为25
            stp.connect(self.mail_host, 25)  
            # set_debuglevel(1)可以打印出和SMTP服务器交互的所有信息
            stp.set_debuglevel(1)
            # 登录邮箱，传递参数1：邮箱地址，参数2：邮箱授权码
            stp.login(self.mail_sender,self.mail_license)
            # 发送邮件，传递参数1：发件人邮箱地址，参数2：收件人邮箱地址，参数3：把邮件内容格式改为str
            stp.sendmail(self.mail_sender, self.mail_receivers, self.mm.as_string())
            print("邮件发送成功")
            # 关闭SMTP对象
            stp.quit()
        except smtplib.SMTPException as e:
            print("error", e)
    
    def change_recerivers(self, new_receivers):
        self.mail_receivers = new_receivers
        
    def send_on_exit(self, *args, **kwargs):
        import atexit
        atexit.register(self.send)
def easymail(filepath):
    import os
    em = EmailSender()
    try :
        with open(filepath, 'r') as f:
            em.get_text(" ".join(f.readlines()))
            em.send_on_exit()
    except Exception as e:
        print(e)
    
# def save_model(model, path, **kwargs):
#     ts = timestamp()
#     save_path = os.path.join(path, ts, )
    
    