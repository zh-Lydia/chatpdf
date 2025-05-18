"""
将查询处理模块集成到原有RAG系统中
"""

import logging
from typing import List, Dict, Any, Tuple
from query_processing import QueryProcessor, IntentType

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-integration")

class EnhancedRAG:
    """增强版RAG系统，集成查询意图识别和查询改写功能"""
    
    def __init__(self, rag_module=None):
        """初始化增强版RAG系统
        
        Args:
            rag_module: 原有RAG模块的实例
        """
        self.rag_module = rag_module
        self.query_processor = QueryProcessor()
        logger.info("初始化增强版RAG系统，集成查询处理功能")
    
    def process_question(self, question: str, enable_web_search: bool = False, model_choice: str = "ollama") -> Tuple[str, List[Dict[str, Any]]]:
        """处理用户问题，集成查询处理和检索增强生成
        
        Args:
            question: 用户问题
            enable_web_search: 是否启用网络搜索
            model_choice: 模型选择
            
        Returns:
            (生成的回答, 使用的资料来源列表)
        """
        # 1. 进行查询处理
        processed = self.query_processor.process_query(question)
        
        # 2. 根据意图决定后续处理
        if not processed['needs_retrieval']:
            # 直接使用模型回答实体类问题
            answer = self._direct_answer(question, processed['intent'])
            return answer, []
        
        # 3. 对于需要检索的问题，使用改写后的查询进行检索
        all_contexts, all_doc_ids, all_metadata = self._enhanced_retrieval(
            processed['rewritten_queries'],
            enable_web_search,
            model_choice
        )
        
        # 4. 使用检索结果生成回答
        answer = self._generate_answer(
            original_question=question,
            retrieval_results=(all_contexts, all_doc_ids, all_metadata),
            intent_type=processed['intent'],
            enable_web_search=enable_web_search,
            model_choice=model_choice
        )
        
        # 5. 构建资料来源
        sources = [{
            'text': doc,
            'metadata': metadata
        } for doc, metadata in zip(all_contexts, all_metadata)]
        
        return answer, sources
    
    def _direct_answer(self, question: str, intent_type: str) -> str:
        """直接使用模型回答不需要检索的问题
        
        Args:
            question: 用户问题
            intent_type: 意图类型
            
        Returns:
            生成的回答
        """
        prompt = f"""用户提出了一个{intent_type}类型的问题，请直接用你的知识回答，无需引用外部资料。
问题：{question}

回答："""
        
        # 调用原有RAG系统中的LLM接口
        # 注意：这里需要根据实际rag_module的实现进行调整
        try:
            # 假设rag_module有一个direct_query方法
            if hasattr(self.rag_module, 'direct_query'):
                return self.rag_module.direct_query(prompt)
            else:
                # 使用本地Ollama模型
                import requests
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "deepseek-r1:7b",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60
                )
                return response.json().get("response", "无法生成回答。")
        except Exception as e:
            logger.error(f"直接回答生成失败: {str(e)}")
            return f"很抱歉，我无法直接回答这个问题。错误信息：{str(e)}"
    
    def _enhanced_retrieval(self, queries: List[str], enable_web_search: bool, model_choice: str) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """使用多个改写查询进行增强检索
        
        Args:
            queries: 改写后的查询列表
            enable_web_search: 是否启用网络搜索
            model_choice: 模型选择
            
        Returns:
            (检索内容列表, 文档ID列表, 元数据列表)
        """
        # 初始化结果集
        all_contexts = []
        all_doc_ids = []
        all_metadata = []
        unique_doc_ids = set()  # 用于去重
        
        # 对每个改写查询进行检索
        for query in queries:
            try:
                # 调用原有RAG系统的检索功能
                # 注意：这里需要根据实际rag_module的实现进行调整
                if hasattr(self.rag_module, 'recursive_retrieval'):
                    contexts, doc_ids, metadata = self.rag_module.recursive_retrieval(
                        initial_query=query,
                        max_iterations=1,  # 由于我们已经手动改写了查询，不需要太多迭代
                        enable_web_search=enable_web_search,
                        model_choice=model_choice
                    )
                    
                    # 去重并添加到结果集
                    for doc, doc_id, meta in zip(contexts, doc_ids, metadata):
                        if doc_id not in unique_doc_ids:
                            unique_doc_ids.add(doc_id)
                            all_contexts.append(doc)
                            all_doc_ids.append(doc_id)
                            all_metadata.append(meta)
                else:
                    logger.warning("原有RAG系统未提供recursive_retrieval方法，无法进行检索")
            except Exception as e:
                logger.error(f"检索查询「{query}」时出错: {str(e)}")
        
        logger.info(f"增强检索完成，共获取 {len(all_contexts)} 个相关文档片段")
        return all_contexts, all_doc_ids, all_metadata
    
    def _generate_answer(self, original_question: str, retrieval_results: Tuple[List[str], List[str], List[Dict[str, Any]]], intent_type: str, enable_web_search: bool, model_choice: str) -> str:
        """基于检索结果生成回答
        
        Args:
            original_question: 原始问题
            retrieval_results: 检索结果（内容，文档ID，元数据）
            intent_type: 问题意图类型
            enable_web_search: 是否启用网络搜索
            model_choice: 模型选择
            
        Returns:
            生成的回答
        """
        all_contexts, all_doc_ids, all_metadata = retrieval_results
        
        # 构建提示词
        context_with_sources = []
        for doc, metadata in zip(all_contexts, all_metadata):
            source_type = metadata.get('source', '本地文档')
            if source_type == 'web':
                url = metadata.get('url', '未知URL')
                title = metadata.get('title', '未知标题')
                context_with_sources.append(f"[网络来源: {title}] (URL: {url})\n{doc}")
            else:
                source = metadata.get('source', '未知来源')
                context_with_sources.append(f"[本地文档: {source}]\n{doc}")
        
        context = "\n\n".join(context_with_sources)
        
        # 根据意图类型调整提示词
        if intent_type == IntentType.BOOLEAN:
            instruction = "请明确给出「是」或「否」的判断，然后解释原因"
        elif intent_type == IntentType.DESCRIPTIVE:
            instruction = "请提供详细、全面的解释，使用适当的标题和段落组织内容"
        else:
            instruction = "请基于参考资料提供准确的回答"
        
        # 组装最终提示词
        prompt_template = f"""作为一个专业的问答助手，你需要基于以下参考资料回答用户的{intent_type}问题。

参考资料：
{context}

用户问题：{original_question}

请遵循以下回答原则：
1. 仅基于提供的参考资料回答问题，不要使用你自己的知识
2. 如果参考资料中没有足够信息，请坦诚告知你无法回答
3. {instruction}
4. 在回答末尾标注信息来源

请现在开始回答："""
        
        # 调用原有RAG系统的生成功能
        try:
            # 使用合适的生成方法
            if hasattr(self.rag_module, 'generate_answer'):
                return self.rag_module.generate_answer(prompt_template, model_choice)
            else:
                # 使用本地Ollama模型
                import requests
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "deepseek-r1:7b" if model_choice == "ollama" else model_choice,
                        "prompt": prompt_template,
                        "stream": False
                    },
                    timeout=120
                )
                return response.json().get("response", "无法生成回答。")
        except Exception as e:
            logger.error(f"回答生成失败: {str(e)}")
            return f"很抱歉，我无法生成回答。错误信息：{str(e)}" 