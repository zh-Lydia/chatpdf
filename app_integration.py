"""
将查询处理功能集成到Gradio界面
"""

import gradio as gr
from query_processing import QueryProcessor, IntentType
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("app-integration")

def integrate_query_processing(original_stream_answer):
    """包装原始stream_answer函数，添加查询处理功能
    
    Args:
        original_stream_answer: 原始的stream_answer函数
        
    Returns:
        增强版的stream_answer函数
    """
    processor = QueryProcessor()
    
    def enhanced_stream_answer(question, enable_web_search=False, model_choice="ollama", progress=None):
        """增强版的stream_answer函数
        
        Args:
            question: 用户问题
            enable_web_search: 是否启用网络搜索
            model_choice: 模型选择
            progress: 进度回调函数
            
        Yields:
            生成的回答和状态
        """
        if progress:
            progress(0.1, desc="分析查询意图...")
        
        # 处理查询
        processed = processor.process_query(question)
        intent_type = processed['intent']
        needs_retrieval = processed['needs_retrieval']
        rewritten_queries = processed['rewritten_queries']
        
        # 提供关于查询处理的反馈
        intent_feedback = f"✓ 查询意图: {intent_type}"
        if needs_retrieval:
            intent_feedback += f"\n✓ 查询改写: 已生成{len(rewritten_queries)-1}个变体"
        else:
            intent_feedback += "\n✓ 直接回答: 无需检索"
        
        yield intent_feedback, "分析查询中..."
        
        if progress:
            progress(0.2, desc="准备检索...")
        
        # 对于实体类查询，直接使用模型回答
        if not needs_retrieval:
            prompt = f"""用户提出了一个{intent_type}类型的问题，请直接用你的知识回答，无需引用外部资料。
问题：{question}

回答："""
            
            import requests
            
            try:
                if progress:
                    progress(0.5, desc="生成直接回答...")
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "deepseek-r1:7b" if model_choice == "ollama" else model_choice,
                        "prompt": prompt,
                        "stream": True
                    },
                    timeout=60,
                    stream=True
                )
                
                answer = intent_feedback + "\n\n"
                
                for line in response.iter_lines():
                    if line:
                        import json
                        chunk = json.loads(line.decode()).get("response", "")
                        answer += chunk
                        yield answer, "生成直接回答中..."
                
                final_answer = answer + "\n\n(此回答由模型直接生成，未使用检索增强)"
                yield final_answer, "完成!"
                
            except Exception as e:
                error_msg = f"生成直接回答时出错: {str(e)}"
                logger.error(error_msg)
                yield intent_feedback + "\n\n" + error_msg, "遇到错误"
                
        else:
            # 对于需要检索的问题，使用原始stream_answer进行处理
            # 但在最终回答中加入意图分析信息
            origin_generator = original_stream_answer(question, enable_web_search, model_choice, progress)
            
            first_response = True
            for response, status in origin_generator:
                if first_response:
                    # 在第一个响应前添加意图分析信息
                    enhanced_response = intent_feedback + "\n\n" + response
                    first_response = False
                else:
                    enhanced_response = intent_feedback + "\n\n" + response
                
                yield enhanced_response, status
    
    return enhanced_stream_answer

def integrate_into_UI(demo, original_stream_answer):
    """将查询处理功能集成到Gradio界面
    
    Args:
        demo: Gradio应用实例
        original_stream_answer: 原始stream_answer函数
        
    Returns:
        修改后的Gradio应用实例
    """
    # 创建增强版的stream_answer函数
    enhanced_stream_answer = integrate_query_processing(original_stream_answer)
    
    # 添加查询处理详情显示区域
    with demo:
        with gr.Tabs():
            with gr.TabItem("💬 问答对话"):
                with gr.Row():
                    with gr.Column(scale=5):
                        query_intent_display = gr.Textbox(
                            label="查询意图分析",
                            value="在提问后显示查询意图分析结果...",
                            lines=2,
                            interactive=False
                        )
    
    # 添加一个函数用于更新UI，显示查询处理结果
    def update_query_info(question):
        processor = QueryProcessor()
        result = processor.process_query(question)
        
        info = f"查询「{question}」的意图类型: {result['intent']} (置信度: {result['confidence']:.2f})\n"
        if result['needs_retrieval']:
            variant_count = len(result['rewritten_queries']) - 1  # 减去原始查询
            info += f"需要检索: 是，已生成{variant_count}个查询变体"
        else:
            info += "需要检索: 否，将直接回答"
        
        return info
    
    # 将原始stream_answer替换为增强版
    import types
    if hasattr(demo, 'stream_answer'):
        demo.stream_answer = enhanced_stream_answer
    
    # 修改process_chat函数以使用enhanced_stream_answer
    if hasattr(demo, 'process_chat'):
        original_process_chat = demo.process_chat
        
        def enhanced_process_chat(question, history, enable_web_search, model_choice):
            # 更新查询意图分析显示
            query_intent_display.update(update_query_info(question))
            
            # 调用原始process_chat函数
            return original_process_chat(question, history, enable_web_search, model_choice)
        
        demo.process_chat = enhanced_process_chat
    
    return demo 