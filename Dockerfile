FROM smina
LABEL maintainer "zyw <643550671@qq.com>"
ADD src /opt/fernie/src
ADD old_src /opt/fernie/old_src
ADD pega_docking /opt/fernie/pega_docking
ADD torch_src /opt/fernie/torch_src
RUN pip3 install python-docx -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install pytorch-lightning -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install easydict -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install "ray[default]" -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install "ray[tune]" -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install ProDy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install numpy protobuf==3.16.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt-get install protobuf-compiler libprotoc-dev -y 
RUN CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON" pip3 install onnx -i https://pypi.tuna.tsinghua.edu.cn/simple
