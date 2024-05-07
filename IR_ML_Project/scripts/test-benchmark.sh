


urls=(
    "https://cloud.tsinghua.edu.cn/f/f9b91bba105a42d29b3c/?dl=1" 
    "https://cloud.tsinghua.edu.cn/f/8388887b25be4feca7d1/?dl=1" 
    "https://cloud.tsinghua.edu.cn/f/99423f19652f454897eb/?dl=1" 
    "https://cloud.tsinghua.edu.cn/f/a5d5fa46ec084bcb8156/?dl=1" 
    "https://cloud.tsinghua.edu.cn/f/f20bb683f2014be6afff/?dl=1" 
    "https://cloud.tsinghua.edu.cn/f/bca4232c7ee9443192fe/?dl=1" 
)

file_names=(
    "cogview2.zip"
    "dalle2.zip"
    "openjourney.zip"
    "sd1-4.zip"
    "sd2-1.zip"
    "versatile-diffusion.zip"
)

len=${

cd benchmark/generations
if [ -z "$(ls -A)" ]; then
    for ((i=0; i<len; i++)); do
        wget -c ${urls[i]} -O ${file_names[i]}
        unzip ${file_names[i]}
        rm ${file_names[i]}
    done
fi
cd ../..


run_cmd="python test-benchmark.py"

echo ${run_cmd}
eval ${run_cmd}
set +x