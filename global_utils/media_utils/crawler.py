import requests
import re
import json
from tqdm import tqdm

youtube_headers = {
    'cookie': 'VISITOR_INFO1_LIVE=9qZVrzB27uI; PREF=f4=4000000&tz=Asia.Shanghai; _ga=GA1.2.621834420.1648121145; _gcl_au=1.1.1853038046.1648121145; NID=511=Zc1APdmEbCD-iqVNVgI_vD_0S3LVI3XSfl-wUZEvvMU2MLePFKsQCaKUlUtchHSg-kWEVMGOhWUbxpQMwHeIuLjhxaslwniMh1OsjVfmOeTfhpwcRYpMgqpZtNQ7qQApY21xEObCvIez6DCMbjRhRQ5P7siOD3X87QX0CFyUxmY; OTZ=6430350_24_24__24_; GPS=1; YSC=0E115KqM_-I; GOOGLE_ABUSE_EXEMPTION=ID=d02004902c3d0f4d:TM=1648620854:C=r:IP=47.57.243.77-:S=YmZXPW7dxbu83bDuauEpXpE; CONSISTENCY=AGDxDeNysJ2boEmzRP4v6cwgg4NsdN4-FYQKHCGhA0AeW1QjFIU1Ejq1j8l6lwAc6c-pYTJiSaQItZ1M6QeI1pQ3wictnWXTOZ6_y8EKlt0Y_JdakwW6srR39-NLuPgSgXrXwtS0XTUGXpdnt4k3JjQ',
    'referer': 'https://www.youtube.com/results?search_query=jk%E7%BE%8E%E5%A5%B3',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36'
}
def get_youtube_video(url):
    response = requests.get(url=url, headers=youtube_headers)
    print(response.text)
    json_str = re.findall('var ytInitialPlayerResponse = (.*?);var', response.text)[0]
    json_data = json.loads(json_str)
    video_url = json_data['streamingData']['adaptiveFormats'][0]['url']
    audio_url = json_data['streamingData']['adaptiveFormats'][-2]['url']
    title = json_data['videoDetails']['title']    
    title = title.replace(' ', '')
    title = re.sub(r'[\/:|?*"<>]', '', title)
    video = requests.get(video_url, stream=True)
    file_size = int(video.headers.get('Content-Length'))
    video_pbar = tqdm(total=file_size)
    with open(f'{title}.mp4', mode='wb') as f:
        # 把视频分成 1024 * 1024 * 2 为等分的大小 进行遍历
        for video_chunk in video.iter_content(1024*1024*2):
            # 写入数据
            f.write(video_chunk)
            # 更新进度条
            video_pbar.set_description(f'正在下载{title}视频中......')
            # 更新进度条长度
            video_pbar.update(1024*1024*2)
        # 下载完毕
        video_pbar.set_description('下载完成！')
        # 关闭进度条
        video_pbar.close() 