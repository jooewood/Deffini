#!/bin/bash
name=001
# 2080ti_max.sh
echo "cd ~/src/fernie/src && bash A40_max_bn.sh > shell_out/${name}_A40_max_bn.out && echo success"
echo "cd ~/src/fernie/src && bash A40_max_relu.sh > shell_out/${name}_A40_max_relu.out && echo success"
echo "cd ~/src/fernie/src && bash A40_max_dp01.sh > shell_out/${name}_A40_max_dp01.out && echo success"
echo "cd ~/src/fernie/src && bash A40_max_relu_bn.sh > shell_out/${name}_A40_max_relu_bn.out && echo success"
echo "cd ~/src/fernie/src && bash A40_max_relu_bn_dp01.sh > shell_out/${name}_A40_max_relu_bn_dp01.out && echo success"
echo "cd ~/src/fernie/src && bash A40_avg.sh > shell_out/${name}_A40_avg.out && echo success"

# pdsh -w n101 "cd ~/src/fernie/src && bash A40_avg_bn.sh > shell_out/${name}_A40_avg_bn.out && echo success"
# pdsh -w n101 "cd ~/src/fernie/src && bash A40_avg_relu.sh > shell_out/${name}_A40_avg_relu.out && echo success"
# pdsh -w n101 "cd ~/src/fernie/src && bash A40_avg_dp01.sh > shell_out/${name}_A40_avg_dp01.out && echo success"
# pdsh -w n101 "cd ~/src/fernie/src && bash A40_avg_relu_bn.sh > shell_out/${name}_A40_avg_relu_bn.out && echo success"
# pdsh -w n101 "cd ~/src/fernie/src && bash A40_avg_relu_bn_dp01.sh > shell_out/${name}_A40_avg_relu_bn_dp01.out && echo success"
