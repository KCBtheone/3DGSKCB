Gaussian-Splatting License  
===========================  

**Inria** and **the Max Planck Institut for Informatik (MPII)** hold all the ownership rights on the *Software* named **gaussian-splatting**.  
The *Software* is in the process of being registered with the Agence pour la Protection des  
Programmes (APP).  

The *Software* is still being developed by the *Licensor*.  

*Licensor*'s goal is to allow the research community to use, test and evaluate  
the *Software*.  

## 1.  Definitions  

*Licensee* means any person or entity that uses the *Software* and distributes  
its *Work*.  

*Licensor* means the owners of the *Software*, i.e Inria and MPII  

*Software* means the original work of authorship made available under this  
License ie gaussian-splatting.  

*Work* means the *Software* and any additions to or derivative works of the  
*Software* that are made available under this License.  


## 2.  Purpose  
This license is intended to define the rights granted to the *Licensee* by  
Licensors under the *Software*.  

## 3.  Rights granted  

For the above reasons Licensors have decided to distribute the *Software*.  
Licensors grant non-exclusive rights to use the *Software* for research purposes  
to research users (both academic and industrial), free of charge, without right  
to sublicense.. The *Software* may be used "non-commercially", i.e., for research  
and/or evaluation purposes only.  

Subject to the terms and conditions of this License, you are granted a  
non-exclusive, royalty-free, license to reproduce, prepare derivative works of,  
publicly display, publicly perform and distribute its *Work* and any resulting  
derivative works in any form.  

## 4.  Limitations  

**4.1 Redistribution.** You may reproduce or distribute the *Work* only if (a) you do  
so under this License, (b) you include a complete copy of this License with  
your distribution, and (c) you retain without modification any copyright,  
patent, trademark, or attribution notices that are present in the *Work*.  

**4.2 Derivative Works.** You may specify that additional or different terms apply  
to the use, reproduction, and distribution of your derivative works of the *Work*  
("Your Terms") only if (a) Your Terms provide that the use limitation in  
Section 2 applies to your derivative works, and (b) you identify the specific  
derivative works that are subject to Your Terms. Notwithstanding Your Terms,  
this License (including the redistribution requirements in Section 3.1) will  
continue to apply to the *Work* itself.  

**4.3** Any other use without of prior consent of Licensors is prohibited. Research  
users explicitly acknowledge having received from Licensors all information  
allowing to appreciate the adequacy between of the *Software* and their needs and  
to undertake all necessary precautions for its execution and use.  

**4.4** The *Software* is provided both as a compiled library file and as source  
code. In case of using the *Software* for a publication or other results obtained  
through the use of the *Software*, users are strongly encouraged to cite the  
corresponding publications as explained in the documentation of the *Software*.  

## 5.  Disclaimer  

THE USER CANNOT USE, EXPLOIT OR DISTRIBUTE THE *SOFTWARE* FOR COMMERCIAL PURPOSES  
WITHOUT PRIOR AND EXPLICIT CONSENT OF LICENSORS. YOU MUST CONTACT INRIA FOR ANY  
UNAUTHORIZED USE: stip-sophia.transfert@inria.fr . ANY SUCH ACTION WILL  
CONSTITUTE A FORGERY. THIS *SOFTWARE* IS PROVIDED "AS IS" WITHOUT ANY WARRANTIES  
OF ANY NATURE AND ANY EXPRESS OR IMPLIED WARRANTIES, WITH REGARDS TO COMMERCIAL  
USE, PROFESSIONNAL USE, LEGAL OR NOT, OR OTHER, OR COMMERCIALISATION OR  
ADAPTATION. UNLESS EXPLICITLY PROVIDED BY LAW, IN NO EVENT, SHALL INRIA OR THE  
AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR  
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE  
GOODS OR SERVICES, LOSS OF USE, DATA, OR PROFITS OR BUSINESS INTERRUPTION)  
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT  
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING FROM, OUT OF OR  
IN CONNECTION WITH THE *SOFTWARE* OR THE USE OR OTHER DEALINGS IN THE *SOFTWARE*.  

## 6.  Files subject to permissive licenses
The contents of the file ```utils/loss_utils.py``` are based on publicly available code authored by Evan Su, which falls under the permissive MIT license. 

Title: pytorch-ssim\
Project code: https://github.com/Po-Hsun-Su/pytorch-ssim\
Copyright Evan Su, 2017\
License: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/LICENSE.txt (MIT)




好的，完全明白。您希望在手动创建环境之后，获得一个完整的命令列表来安装environment.yml文件中指定的所有依赖项。

这个策略非常好，因为它避免了conda env create命令的复杂求解过程，而是将依赖安装分解为更可控的步骤。

以下是您在手动创建并激活gaussian_splatt-ing环境之后，需要执行的所有命令。

Windows环境下手动安装3DGS依赖的完整命令集

前提： 您已经按照之前的指南，成功执行了以下两步：

conda create --name gaussian_splatting python=3.7 -y

conda activate gaussian_splatting

现在，请在激活了(gaussian_splatting)环境的Anaconda Prompt中，逐一执行以下命令块。

第一步：配置镜像源 (如果还没做)

为了确保下载速度和成功率，请先运行这些命令。

code
Cmd
download
content_copy
expand_less

REM --- 配置Conda和Pip镜像源 (推荐) ---
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
第二步：安装核心的Conda包

environment.yml文件指定了非常精确的版本。我们将严格按照这些版本进行安装，这能最大程度保证兼容性。

code
Cmd
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
REM --- 安装PyTorch, Torchvision, Torchaudio和CUDA Toolkit ---
REM 我们明确指定从pytorch channel安装，并指定版本号。
REM 注意：官方yml指定的是cudatoolkit=11.6，但PyTorch 1.12.1通常与CUDA 11.6或11.7配合。我们会优先尝试11.6。
conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.6 -c pytorch -y

REM --- 安装其他Conda包 ---
REM 我们明确指定版本号。
conda install plyfile tqdm pip=22.3.1 -c conda-forge -y

注意： 如果cudatoolkit=11.6找不到包，Conda可能会提示你使用其他版本，比如11.7，通常这也是可以接受的。

第三步：安装Pip包 (来自requirements.txt和yml文件)

environment.yml文件中有一部分包是通过pip安装的。

code
Cmd
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
REM --- 安装yml文件中指定的其他pip包 ---
pip install opencv-python joblib
第四步：进入项目目录并编译自定义CUDA内核 (最关键)

现在，所有基础依赖都已安装完毕，我们可以开始编译3DGS自己的CUDA模块了。

进入gaussian-splatting主目录：

请确保将路径替换为您自己的实际路径。

code
Cmd
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
cd C:\Users\HP\Desktop\code\ind_test\gaussian-splatting

编译diff-gaussian-rasterization：

code
Cmd
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
pip install submodules\diff-gaussian-rasterization

编译simple-knn：

code
Cmd
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
pip install submodules\simple-knn

(可选) 编译 fused-ssim

这个模块主要用于渲染质量的评估，对于核心的三维重建和变化检测不是必需的。如果编译失败可以暂时跳过。

code
Cmd
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
pip install submodules\fused-ssim

重要提示： 编译这几个子模块时，您的终端必须处于激活的(gaussian_splatting)环境下，并且您的电脑上必须已经正确安装了Visual Studio 2019和CUDA Toolkit。

第五步：最终验证

完成所有安装后，运行以下命令来确认一切正常。

code
Cmd
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); import diff_gaussian_rasterization; print('diff_gaussian_rasterization imported successfully!')"

如果您能看到正确的PyTorch版本（1.12.1）、CUDA available: True、以及diff_gaussian_rasterization成功导入的消息，那么您的环境就完全配置成功，可以开始进行三维重建了。

这个命令列表为您提供了一条清晰、可控的安装路径，大大降低了因Conda求解器冲突而导致失败的风险。