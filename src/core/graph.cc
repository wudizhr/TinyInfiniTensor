#include "core/graph.h"
#include "operators/matmul.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        Operator last_opt;
        // OpVec delop;
        // TensorVec save_tensors;
        for(size_t i = 0; i < ops.size();)
        {
            auto op = ops[i];
            if(op->getOpType().underlying() == 10)
            {
                auto predecessors = op->getPredecessors();
                for(auto predecessor : predecessors)
                {
                    if(predecessor->getOpType().underlying() == 10)
                    {
                        TensorVec Outputs = op->getOutputs();
                        TensorVec Inputs = predecessor->getInputs();
                        Tensor input = Inputs[0];
                        Tensor output = Outputs[0];
                        if(output->getDims() == input->getDims())
                        {
                            std::cout << "transpose delete" << std::endl;
                            this->removeOperator(op);
                            this->removeOperator(predecessor);
                            this->removeTensor(op->getInputs()[0]);
                            this->removeTensor(op->getOutputs()[0]);

                            if (auto pred = input->getSource())
                            {
                                pred->removeSuccessors(predecessor);
                                for (auto &succ : output->getTargets())
                                {
                                    pred->addSuccessors(succ);
                                }
                            }
                            input->removeTarget(predecessor);
                            for (auto &succ : output->getTargets())
                            {
                                input->addTarget(succ);
                                succ->replaceInput(output, input);
                                succ->removePredecessors(op);
                                for (auto &predop : predecessor->getPredecessors())
                                {
                                    succ->addPredecessors(predop);
                                }
                            }
                            i--;
                            continue;
                        }
                    }
                }                
            }
            else if(op->getOpType().underlying() == 7)
            {
                Tensor tensorA = op->getInputs()[0];
                Tensor tensorB = op->getInputs()[1];
                if(const auto &source = tensorA->getSource())
                {
                    if(source->getOpType().underlying() == 10)
                    {
                        Tensor input = source->getInputs()[0];
                        Tensor output = source->getOutputs()[0];
                        auto input_dim = input->getDims();
                        auto output_dim = output->getDims();
                        if(input_dim[input_dim.size()-1] == output_dim[output_dim.size()-2])
                        {
                            std::cout << "transpose merge A" << std::endl;
                            Tensor input = source->getInputs()[0];
                            Tensor output = source->getOutputs()[0];
                            
                            // update op info
                            for (auto &predop : source->getPredecessors())
                            {
                                predop->removeSuccessors(source);
                                predop->addSuccessors(op);
                                op->removePredecessors(source);
                                op->addPredecessors(predop);
                            }  
                            input->removeTarget(source);
                            input->addTarget(op);
                            op->replaceInput(output, input);   
                            auto* matmulOp = dynamic_cast<MatmulObj*>(source.get());
                            matmulOp->setTransA(true);
                            continue;                  
                        }
                    }
                }
                if(const auto &source = tensorB->getSource())
                {
                    if(source->getOpType().underlying() == 10)
                    {
                        Tensor input = source->getInputs()[0];
                        Tensor output = source->getOutputs()[0];
                        auto input_dim = input->getDims();
                        auto output_dim = output->getDims();
                        if(input_dim[input_dim.size()-1] == output_dim[output_dim.size()-2])
                        {
                            std::cout << "transpose merge B" << std::endl;
                            // std::cout << input << std::endl;
                            Tensor input = source->getInputs()[0];
                            Tensor output = source->getOutputs()[0];
                            // update op info
                            op->removePredecessors(source);
                            for (auto &predop : source->getPredecessors())
                            {
                                predop->removeSuccessors(source);
                                predop->addSuccessors(op);
                                op->addPredecessors(predop);
                            }  
                            input->removeTarget(source);
                            input->addTarget(op);
                            op->replaceInput(output, input);  
                            auto* matmulOp = dynamic_cast<MatmulObj*>(op.get());
                            matmulOp->setTransB(true);
                            this->removeOperator(source);
                            this->removeTensor(output);
                            // this->print();
                            continue;                  
                        }
                    }
                }
            }
            i++;
        }
        std::cout << "Optimize complete!" << std::endl << std::endl;

        // =================================== 作业 ===================================
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // Naive Version
        std::unordered_map<std::shared_ptr<infini::TensorObj>, size_t> tensorToOffset;
        for(auto tensor : tensors)
        {
            tensorToOffset[tensor] = allocator.alloc(tensor->getBytes());
            // std::cout << "loop1end" << std::endl;
        }
        for(auto tensor : tensors)
        {
            tensor->setDataBlob(make_ref<BlobObj>
                (
                    tensor->runtime,
                    static_cast<uint8_t *>(allocator.getPtr()) +
                        tensorToOffset[tensor]
                )
            );
        }
        // =================================== 作业 ===================================

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini