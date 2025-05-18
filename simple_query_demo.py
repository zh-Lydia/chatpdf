"""
简单的终端查询处理演示程序 - 演示查询改写和意图识别功能
"""

from query_processing import QueryProcessor

def main():
    print("="*70)
    print("查询处理演示程序 - 查询改写与意图识别")
    print("="*70)
    print("本程序将演示如何对用户输入的查询进行意图识别和查询改写")
    print("输入'q'或'退出'可结束程序")
    print("-"*70)
    
    # 初始化查询处理器
    processor = QueryProcessor()
    
    while True:
        # 获取用户输入
        query = input("\n请输入您的查询：")
        
        # 检查是否退出
        if query.lower() in ['q', 'quit', '退出', 'exit']:
            print("\n感谢使用！再见！")
            break
            
        if not query.strip():
            print("查询不能为空，请重新输入")
            continue
        
        # 处理查询
        result = processor.process_query(query)
        
        # 输出查询处理结果
        print("\n" + "="*50)
        print("【查询处理结果】")
        print(f"原始查询: {result['original_query']}")
        print(f"意图类型: {result['intent']} (置信度: {result['confidence']:.2f})")
        print(f"需要检索: {'是' if result['needs_retrieval'] else '否'}")
        
        # 如果有查询改写，显示改写结果
        if result['needs_retrieval'] and result['rewritten_queries']:
            print("\n【查询改写结果】")
            for i, variant in enumerate(result['rewritten_queries']):
                print(f"  {i+1}. {variant}")
        print("="*50)

if __name__ == "__main__":
    main() 