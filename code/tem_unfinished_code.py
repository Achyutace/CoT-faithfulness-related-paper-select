import os
import pandas as pd
import json
import re

from openai import OpenAI

def api_response(title, abstract):
    if not abstract or len(abstract) < 20:
        return {"title": title, "related_score": "Unknown", "type": "Unknown", "domain": "Unknown", "tagging_reasoning": "摘要缺失"}
    system_instruction = f"""

    """
    user_content = f"""

    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        
        # 使用正则提取标签内容
        match = re.search(r'<JSON_OUTPUT>(.*?)</JSON_OUTPUT>', content, re.DOTALL | re.IGNORECASE)
        if match:
            result = json.loads(match.group(1).strip())
        else:
            # 降级处理
            json_str = content.replace('```json', '').replace('```', '').strip()
            result = json.loads(json_str)
            
        result["title"] = title
        return result
    except Exception as e:
        return {"title": title, "related_score": "Error", "type": "Error", "domain": "Error", "tagging_reasoning": str(e)}

