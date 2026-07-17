# <p align="center">` SmartCutter ` </p>

<p align="center"> ㅤㅤ👇 You can join this discord server below ( RVC / AI Audio friendly ) 👇ㅤㅤ </p>

</p>
<p align="center">
  <a href="https://discord.gg/ymfdwx5jwZ" target="_blank"> Codename's Sanctuary</a>
</p>

<p align="center"> ㅤㅤ👆 To stay up-to-date with advancements, hang out or get support 👆ㅤㅤ </p>

## <p align="center"> A lil bit more about the project:

### <p align="center"> Machine Learning based silence-truncation. <br/> Made with Applio / RVC and my Codename-RVC-Fork-4 in mind. ✨ <br/>
### What's new:
- Supports any sample rate
- Might be better?

ㅤ
<br/>
# ⚠️ㅤ**IMPORTANT** ㅤ⚠️
- For now only CUDA ( nvidia ) or CPU.
- Supportes all sample rates<br/>
- Silence / Sub-Silence ( noisy ) spacings below 100ms are ignored / not processed by design.
- There are limits, it is still a very-low-noise or pure silence focused truncator.
( So keep in mind models might hiccup on some really hard cases. ) <br/>
<br/>
 
✨ to-do list ✨
> - Better pretrained models
 
💡 Ideas / concepts 💡
> - Currently none. Open to your ideas ~
 
 
### ❗ For contact, please join the discord server ❗
 <br/>
 
## Getting Started:

### INSTALLATION:

Run the installation script:

- Double-click `install.bat`.
 
### PRETRAINED MODELS:

- Download the checkpoint <br/>
[v6_model](https://huggingface.co/Razer112/smartcutter-omi/resolve/main/v6_omi.pth?download=true)<br/>
- Put them in SmartCutter's "ckpts" folder
 
### INFERENCE:
 
To start inference:
- First put the concatenated sample or samples ( .wav or .flac ) into "infer_input" dir.
- Double-click `run-infer.bat`.
- Results will land in "infer_output" dir.<br/>
`( Concatenated = Simply join up all samples / segments into 1 file )`<br/><br/>`NOTE: supports multiple samples AND multiple sr.`
 
### TRAINING:
- Training of custom pretrains is supported. <br/> Instruction regarding that will be published in future.
 
