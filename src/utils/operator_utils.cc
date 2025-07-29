#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    std::vector<int> nB(B);
    std::vector<int> nA(A);
    if(A.size() > B.size())
    {
        for(size_t i=0;i<(A.size()-B.size());i++)
        {
            nB.insert(nB.begin(), 1); 
            // std::cout << "B" << std::endl;
        }
            
    }
    if(B.size() > A.size())
    {
        for(size_t i=0;i<(B.size()-A.size());i++)
        {
            nA.insert(nA.begin(), 1); 
            // std::cout << "A" << std::endl;
        }
            
    }
    for(size_t i=0;i<nA.size();i++)
    {
        if(nA[i]==1)
        {
            nA[i] = nB[i];
        }
    }
    // std::cout << "A:";
    // for(size_t i=0;i<A.size();i++)
    // {
    //     std::cout << A[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "B:";
    // for(size_t i=0;i<B.size();i++)
    // {
    //     std::cout << B[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "OUTPUT:";
    // for(size_t i=0;i<nA.size();i++)
    // {
    //     std::cout << nA[i] << " ";
    // }
    // std::cout << std::endl;
    // =================================== 作业 ===================================
    return {nA};
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
