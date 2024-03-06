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
import os, yaml

def torch_log_decorator(func):
    """decorator for torch log

    Args:
        func : function to be decorated, should return a dict with key "loss" and "accuracy"
    """
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
    """mkdir if not exists

    python's mkdir.
    
    Args:
        path (str): path to be mkdir
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_time_str(year_first=True):
    """An easy way to get time string

    Returns:
        str : time string
    """
    from datetime import datetime
    import pytz
    
    now = datetime.now(pytz.timezone('Asia/Shanghai'))
    if year_first:
        timestamp = now.strftime("%y%m%d%H%M%S")
    else:
        timestamp =now.strftime("%H%M%S_%d%m%y")
    return timestamp
    

class Logger(object):
    """Logger utils to save log to a file 

    stdout will first be redirected to this logger, then the log will be saved to a file. the `start_capture` and `stop_capture` function is used to control the redirection. 
    """
    def __init__(self, logFile=None):
        """init function, set the log file path

        Args:
            logFile (str, optional): the file path for the saved log. Defaults to './Default.log'.
        """
        self.terminal = sys.stdout
        self.log = ''
        self.close_flag = False
        self.logFile = logFile or os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'log/', get_time_str() + '.log')
        open(self.logFile, 'w').close
    
    def start_capture(self):
        """start capture the stdout
        """
        sys.stdout = self
        
    def stop_capture(self):
        """stop capture the stdout
        """
        sys.stdout = self.terminal
        self.close_flag = True
        with open(self.logFile, 'w') as f:
            f.write(self.log)
    
    def write(self, message):
        """write function, write the message to stdout and log file

        Args:
            message (_type_): _description_
        """
        self.terminal.write(message)
        if self.close_flag is False:
            self.log += message
    
    def flush(self):
        """flush function, flush the log file
        """
        self.terminal.flush()

def ez_logger(path=None):
    if path == None:
        path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'log/', get_time_str() + '.log')
    log = Logger(path)
    log.start_capture()

class EmailSender:
    """Email Sender, send text, picture and attachment
    
    An Email sender function is designed to send information about the training process and the training results to the user after the training is completed, allowing the user to promptly receive a signal indicating the completion of the training. 
    """
    def __init__(self, title="Message", config_path=None):
        """init function, set some default values

        
        Args:
            title (str, optional): the title of the email. Defaults to "Message".
        """
        if config_path is None:
            if os.path.exists("./email_config.yaml"):
                config_path = "./email_config.yaml"
            elif os.path.exists("../email_config.yaml"):
                config_path = "../email_config.yaml"   
            elif os.path.exists("../../email_config.yaml"):
                config_path = "../../email_config.yaml"
        config = read_yaml_to_dict(config_path)
        
        self.mail_host = config["mail_host"]
        self.mail_sender = config["mail_sender"]
        self.mail_license = config["mail_license"]
        self.mail_receivers = config["mail_receivers"]
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
        """get text, add it to email

        Args:
            text (_type_): text to be sent
        """
        # 邮件正文内容
        body_content = text
        # 构造文本,参数1：正文内容，参数2：文本格式，参数3：编码方式
        message_text = MIMEText(body_content,"plain","utf-8")
        # 向MIMEMultipart对象中添加文本对象
        self.mm.attach(message_text)
    
    def get_picture(self, image_data):
        """get picture, add it to email

        Args:
            image_data (_type_): image data to be sent
        """
        # 二进制读取图片
        # image_data = open('a.jpg','rb')
        # 设置读取获取的二进制数据
        message_image = MIMEImage(image_data.read())
        # 关闭刚才打开的文件
        image_data.close()
        # 添加图片文件到邮件信息当中去
        self.mm.attach(message_image)
        
    def get_attachment(self, attachment_path):
        """get attachment, add it to email

        Args:
            attachment_path (_type_): attachment path to be sent
        """
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
        """change receivers

        Args:
            new_receivers (str or List[str]): new receivers
        """
        self.mail_receivers = new_receivers
        
    def send_on_exit(self, *args, **kwargs):
        """send email on exit
        
        Args:
            *args: args
        """
        import atexit
        atexit.register(self.send)
        

def easymail(filepath):
    """send email with text in file

    Args:
        filepath (str): the file path
    """
    import os
    em = EmailSender()
    try :
        with open(filepath, 'r') as f:
            em.get_text(" ".join(f.readlines()))
            em.send_on_exit()
    except Exception as e:
        print(e)    

def args_decorate(path, show=False):
    """a decorator to save args to yaml file

    Args:
        path (str): yaml file path
        show (bool, optional): show args or not. Defaults to False.

    Examples:
        >>> class A:
        >>> @yaml_decorate('./config.yaml')
        >>> def __init__(self, a, b, c):
        >>>     pass
    
    """
    import yaml
    import functools
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            import inspect 
            func_signature = inspect.signature(func)
            func_params = list(func_signature.parameters.keys())

            # Filter out parent class parameters from args
            filtered_args = [arg for i, arg in enumerate(args) if func_params[i] not in kwargs]
            if func_params[0] == 'self':
                filtered_args = filtered_args[1:]
            params = {
                'args': filtered_args,
                'kwargs': kwargs,
            }
            if show:
                print('args', filtered_args)
                for k, v in kwargs.items():
                    print(k, v)

            # 将参数字典保存为 YAML 文件
            with open(path, 'w') as file:
                yaml.dump(params, file)

            return result
        return wrapper
    return decorator


    
    
def save_dict_to_yaml(dict_value, save_path):
    """save dict to yaml file

    Args:
        dict_value (dictionary): dict to be saved
        save_path (str): yaml file path
    """
    with open(save_path, 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))

        
def read_yaml_to_dict(yaml_path):
    """read yaml file to dict

    Args:
        yaml_path (str): yaml file path

    Returns:
        dict: dict
    """
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


def boolean_string(s):
    """Convert a string into a boolean value.   

    Args:
        s (str): string to be converted

    Raises:
        ValueError: Not a valid boolean string

    Returns:
        bool: boolean valueleish
    """
    s = s.lower().strip()
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def train_log(log_name=None):
    """a decorator to save train log

    A decorate to save train log to log file. The log file will be saved in the 

    Args:
        log_name (str, optional): log file name, the log would save to `os.path.join(main_path, 'log/', log_name)`. Defaults to None.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # the path of the main file
            current_log_name = log_name or get_time_str()
            main_path = os.path.dirname(os.path.abspath(sys.argv[0]))
            log_path = os.path.join(main_path, 'log/', current_log_name)
            mkdir(log_path)
            log_file = 'train_log.txt'
            log = Logger(os.path.join(log_path, log_file))
            log.start_capture()
            result = func(*args, **kwargs)
            log.stop_capture()
            # easymail(os.path.join(log_path, log_file))
            return result
        return wrapper
    return decorator
        