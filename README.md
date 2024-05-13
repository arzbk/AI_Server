# Conversational AI Server

The goal of this project is to provide a basic, secure (with valid HTTPS certificates) server for running various AI tasks on a remote computer or VM. Most computers do not have the resources to run these types of computations, so 
using this project, you can out-source this work to a computer that does.

The computer that runs this should ideally have at least 12GB of VRAM, and CUDA installed. Depending on the models used, this requirement could balloon to over 48GB of VRAM required.

When this is complete, the websocket endpoint will allow you to talk to the computer, and your voice will be converted to text, fed to the LLM, the LLM will reply conversationally, and the text to speech model will give the computer a 
voice to respond with - enabling you to have full conversations with your PC.
