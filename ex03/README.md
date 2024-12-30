## OCR based webpage parsing

### 1. Find relevant products from the video recorded session

For this I used the following tools and open source libraries, which can be installed as 
follows:
```commandline
apt install ffmpeg
pip install scenedetect
pip install paddlepaddle
pip install trankit
```

Once installed, the script `run.sh` provided can be executed. 
The shell script has comments describing the actions of commands.
After the execution of `run.sh`, we can obtain the relevant products discussed in the 
screen recording session using:
```commandline
# Filter output based on NER S-ORG
cat output/tokens.json|grep S-ORG|cut -d' ' -f2|sort |uniq
```
This will have some false alarms but in my test, it creates no mis-detections.

### 2. Broaden the context around the found products in terms of information like price, related other products etc.

I spent around 5 hours to design the first part. For the second part, one can perform the 
following:

#### 1. Get coarse weights for every scene from input video, select top 5:
```commandline
tail -n +3 home_gym-Scenes.csv|cut -d, -f1,8|sort -k 2,2 -rn -k 1,1|head -5
20,1409
8,1587
18,173
17,363
16,349
```
#### 2. From these, find relevant bounding box information 
```commandline
grep -e 'home_gym-Scene-020\|home_gym-Scene-008' analysis.log
...
home_gym-Scene-008-02.jpg [1199, 480, 1594, 480, 1594, 497, 1199, 497] Rogue Flat Utility Bench 2.0-Premium Textured Pad +$195 
home_gym-Scene-008-02.jpg [1186, 519, 1250, 524, 1249, 543, 1184, 538] Add Plates
home_gym-Scene-008-02.jpg [485, 547, 564, 551, 563, 570, 484, 566] 5/8Hardware
home_gym-Scene-008-02.jpg [1197, 553, 1388, 554, 1388, 577, 1197, 575] 260LB HG2.0 Set (+$595.00)
home_gym-Scene-008-02.jpg [1355, 614, 1439, 619, 1438, 638, 1354, 634] Add to Cart
home_gym-Scene-008-02.jpg [1185, 670, 1561, 672, 1561, 694, 1185, 692] ROGUE HR-2 HALF RACK ACCESSORIES Available Options (3)
home_gym-Scene-008-02.jpg [84, 705, 214, 706, 213, 725, 84, 724] Product Description
home_gym-Scene-008-02.jpg [240, 708, 317, 708, 317, 725, 240, 725] Gear Specs
...
```
#### 3. Parsing these we can get price information