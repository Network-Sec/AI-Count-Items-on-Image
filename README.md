# Count Items on Image using AI / ML model
Modern AI / ML based counter for items in an image. 

As usual, some TODOs remaining, recognition isn't yet 100%, that is highly `model` dependend. 

## Install
```bash
$ git clone https://github.com/Network-Sec/AI-Count-Items-on-Image.git
$ cd AI-Count-Items-on-Image
$ git submodule update --init --recursive
```

## Usage
- Install deps
- Run and wait for model download (may need manual downloads)
- Go to http://127.0.0.1:5000/ and upload image

Results will not be perfect and vary depending on model, weights and other settings. 

We made a table output below main image to show results of different combos - this isn't fully finished yet. 

Final goal would be to add a DB and have some preprocessing (we already do but very generic, should be image specific), weight adjustment etc. 
But ATM it was good enough for our purpose - you could `easily wrap an Android App` around this server-side functionality... 

...and maybe, just maybe, people would `send you images` with `"things" to count` XD

## V2 - WIP
Not yet released, current version screenshot below

![AI_Count_WIP_V2](https://github.com/user-attachments/assets/a0207f45-2975-4702-8fb9-cc1d334667ec)

## Current Version Screenshot
![Example Result](https://github.com/user-attachments/assets/9b0d75b7-d5d7-40d1-a4cc-541a210329ee)

Inspired by the **Intelligence Community**. 
