"""
查询处理模块 - 包含查询意图识别和查询改写功能
"""

import logging
import json
import re
from typing import List, Dict, Tuple, Any
import requests
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("query-processing")

# 从环境变量获取API配置
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("QUERY_PROCESSING_MODEL", "deepseek-r1:1.5b")

# 意图类型定义
class IntentType:
    ENTITY = "实体"            # 查询单一实体的问题
    DESCRIPTIVE = "描述性"      # 需要详细描述或解释的问题
    BOOLEAN = "是非类"         # 可以用是/否回答的问题
    UNKNOWN = "未知"           # 无法确定类型的问题

class QueryProcessor:
    """查询处理器，提供意图识别和查询改写功能"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        """初始化查询处理器
        
        Args:
            model_name: 使用的模型名称
        """
        self.model_name = model_name
        logger.info(f"初始化查询处理器，使用模型: {model_name}")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """处理用户查询，包括意图识别和查询改写
        
        Args:
            query: 用户原始查询
            
        Returns:
            包含处理结果的字典
        """
        # 1. 进行意图识别
        intent, confidence = self.identify_intent(query)
        
        # 2. 根据意图决定是否进行查询改写
        rewritten_queries = []
        needs_retrieval = False
        
        if intent == IntentType.ENTITY:
            # 实体类查询不需要检索
            needs_retrieval = False
            logger.info(f"查询「{query}」被识别为实体查询，跳过检索")
        else:
            # 描述性和是非类查询需要检索
            needs_retrieval = True
            # 只有需要检索的查询才进行查询改写
            rewritten_queries = self.rewrite_query(query)
            logger.info(f"查询「{query}」被识别为{intent}查询，将进行检索")
        
        return {
            "original_query": query,
            "intent": intent,
            "confidence": confidence,
            "needs_retrieval": needs_retrieval,
            "rewritten_queries": rewritten_queries,
        }
    
    def identify_intent(self, query: str) -> Tuple[str, float]:
        """识别查询的意图类型
        
        Args:
            query: 用户查询
            
        Returns:
            (意图类型, 置信度)
        """
        prompt = f"""作为一个高级查询意图分类器，请分析以下用户查询的意图类型。
将查询分为以下三种类型之一:
1. 实体: 用户在查询特定的人、地点、物体或概念的基本信息。例如"钠是什么"、"北京在哪里"、"牛顿是谁"。
2. 描述性: 用户需要详细解释、过程描述或复杂概念的阐述。例如"如何制作蛋糕"、"为什么天空是蓝色的"、"Python的主要特性"。
3. 是非类: 用户提出可以用"是"或"否"回答的问题。例如"地球是圆的吗"、"熊猫会游泳吗"、"Python比Java难学吗"。

用户查询: {query}

请首先详细分析查询的特点和结构，然后给出你的分类结果。
回答格式: {{"分析": "...", "类型": "实体|描述性|是非类", "置信度": 0.1-1.0}}
"""
        try:
            response = self._call_model(prompt)
            # 解析JSON格式的回答
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                intent_type = result.get("类型", IntentType.UNKNOWN)
                confidence = float(result.get("置信度", 0.5))
                return intent_type, confidence
            else:
                logger.warning(f"模型响应格式不符合预期: {response}")
                # 使用简单的规则作为备选方案
                return self._rule_based_intent(query), 0.5
        except Exception as e:
            logger.error(f"意图识别过程出错: {str(e)}")
            return self._rule_based_intent(query), 0.3
    
    def _rule_based_intent(self, query: str) -> str:
        """基于简单规则的意图识别备选方案
        
        Args:
            query: 用户查询
            
        Returns:
            意图类型
        """
        # 是非类问题常见结尾
        if re.search(r'吗\s*[?？]*$|是不是|能否|能不能|可不可以', query):
            return IntentType.BOOLEAN
        
        # 实体类问题常见句式
        if re.match(r'^[^是]+是[什么谁]+|^什么是|^谁是|在哪里|^谁[发明创造]了', query):
            return IntentType.ENTITY
            
        # 默认为描述类
        return IntentType.DESCRIPTIVE
    
    def rewrite_query(self, query: str, num_variations: int = 3) -> List[str]:
        """使用零样本学习方法改写查询，生成多样化变体以提高检索效果
        
        Args:
            query: 原始查询
            num_variations: 生成的改写变体数量
            
        Returns:
            改写后的查询列表
        """
        prompt = f"""作为一个专业的查询改写系统，请帮我改写以下查询，生成{num_variations}个不同的变体，以便提高搜索效果。

原始查询: {query}

请考虑以下改写策略:
1. 同义词替换: 用同义词替换查询中的关键词
2. 专业术语替换: 如适用，用更专业/更通俗的术语替换
3. 问题重构: 以不同方式重新表述问题
4. 扩展查询: 添加潜在相关的内容或限定词

回答格式:
{{"变体": [
  "变体1",
  "变体2",
  "变体3"
]}}

请确保每个变体都保持原始查询的核心意图，但表达方式或用词有所不同。
"""
        try:
            response = self._call_model(prompt)
            # 解析JSON格式的回答
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                variants = result.get("变体", [])
                # 去重并过滤空字符串
                unique_variants = list(set([v.strip() for v in variants if v.strip()]))
                logger.info(f"成功生成 {len(unique_variants)} 个查询变体")
                # 添加原始查询作为第一个变体
                if query not in unique_variants:
                    unique_variants.insert(0, query)
                return unique_variants
            else:
                logger.warning(f"模型响应格式不符合预期: {response}")
                return [query]  # 返回原始查询
        except Exception as e:
            logger.error(f"查询改写过程出错: {str(e)}")
            return [query]  # 出错时返回原始查询
    
    def _call_model(self, prompt: str) -> str:
        """调用LLM模型生成回答
        
        Args:
            prompt: 提示词
            
        Returns:
            模型生成的回答
        """
        try:
            # 使用Ollama本地API
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"调用模型API失败: {str(e)}")
            raise

# 使用示例
if __name__ == "__main__":
    processor = QueryProcessor()
    
    # 测试不同类型的查询
    test_queries = [
        "什么是人工智能",  # 实体类
        "如何使用Python进行数据分析",  # 描述性
        "地球是平的吗",  # 是非类
        "为什么天空是蓝色的",  # 描述性
        "西红柿炒鸡蛋的做法",  # 描述性
        "谁发明了电灯泡",  # 实体类
    ]
    
    for query in test_queries:
        result = processor.process_query(query)
        print("\n" + "="*50)
        print(f"原始查询: {result['original_query']}")
        print(f"意图类型: {result['intent']} (置信度: {result['confidence']:.2f})")
        print(f"需要检索: {'是' if result['needs_retrieval'] else '否'}")
        
        if result['needs_retrieval'] and result['rewritten_queries']:
            print("改写变体:")
            for i, variant in enumerate(result['rewritten_queries']):
                print(f"  {i+1}. {variant}") 