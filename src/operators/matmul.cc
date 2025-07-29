#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        const auto A = inputs[0];
        const auto B = inputs[1];
        Shape dimA = A->getDims();
        Shape dimB = B->getDims();
        Shape out_shape = dimA;
        int tempA = dimA.size() - 1;
        int tempB = dimA.size() - 1;
        if(dimA[dimA.size()-1] == dimB[dimB.size()-1])
        {
            tempA = dimA.size()-2;
            tempB = dimB.size()-2;
        }
        else if(dimA[dimA.size()-1] == dimB[dimB.size()-2])
        {
            tempA = dimA.size()-2;
            tempB = dimB.size()-1;            
        }
        else if(dimA[dimA.size()-2] == dimB[dimB.size()-1])
        {
            tempA = dimA.size()-1;
            tempB = dimB.size()-2;            
        }
        else if(dimA[dimA.size()-2] == dimB[dimB.size()-2])
        {
            tempA = dimA.size()-1;
            tempB = dimB.size()-1;            
        }
        out_shape[dimA.size()-2] = dimA[tempA];
        out_shape[dimA.size()-1] = dimB[tempB];
        // =================================== 作业 ===================================
        return {{out_shape}};
    }

} // namespace infini