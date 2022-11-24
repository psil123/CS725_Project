# Drowsiness Detection (CS725 Project)

## Team Members 
Pritam Sil - 21Q05R001 <br>
Dishank Aggarwal - 21Q05R003 <br>
Vivek Anil Pandey - 21Q05R004 <br>
Sai Sunil Chothave - 21Q05R005 <br>

## Instructions

1. Download Weights from here : [Google Drive](https://drive.google.com/drive/folders/1OU4bsfZFdOrAfTuTT3r0qdUavllQYgIw?usp=share_link)
2. Sample videos can be downloaded from here : [Google Drive](https://drive.google.com/drive/folders/1OU4bsfZFdOrAfTuTT3r0qdUavllQYgIw?usp=share_link)
3. Save the weights in the same folder as demo.py
4. Run as ` python3 demo.py`
5. Various options to run this script
<pre><code>
usage: demo.py [-h] [-i INPUT] [-t TYPE] [-r ROTATE] [-m {Resnet101,VGG16,MobileNet}]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input 0 for webcam and file name for a video/image
  -t TYPE, --type TYPE  Read from Video(v) or Image(i)
  -r ROTATE, --rotate ROTATE
                        Rotate image by 180 or not
  -m {Resnet101,MobileNet}, --model {Resnet101,MobileNet}
                        Which model to use
</code></pre>

