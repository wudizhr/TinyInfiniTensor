#include "operators/concat.h"
#include "utils/operator_utils.h"
#include "core/graph.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
        // TensorVec a = {nullptr};
        // std::cout << "outputsize:"  << a.size() << std::endl; //这个空元素竟然也会size+1
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph)); //outputs established in here
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    Shape ans = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    for(size_t i = 0;i < rank;i++)
    {
        for(size_t j = 1;j < inputs.size();j++)
        {
            // inputs[j]->print();
            if(dims[i] != inputs[j]->getDims()[i])
            {
                ans[i] += inputs[j]->getDims()[i];
            }     
        }
    }
    // =================================== 作业 ===================================

    return {{ans}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
