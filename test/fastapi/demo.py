import asyncio

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse, FileResponse, PlainTextResponse, HTMLResponse, RedirectResponse, \
    StreamingResponse, Response

# 1. 创建一个FastAPI应用程序的示例对象
app = FastAPI()

# 2. 创建一个路由处理函数
@app.get("/", summary="第一个测试")
async def index():
    """
    这是第一个测试函数
    """

    print("index")
    return {"message": "Hello World22222333333"}

# 3. 路径参数和查询字符串参数的测试
# 访问 http://127.0.0.1:8000/items/5?q=somequery
# item_id: 路径参数 (自动转为 int)
# q: 查询字符串参数 (可选，默认 None)
@app.get("/items/{item_id}", summary="获取指定参数")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

# 4. 分页案例
# 接收? skip=? & limit = ? (查询字符串参数)
@app.get("/items/", summary="分页")
async def read_item(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}


# 5. 使用pydantic
# 5.1 定义数据模型
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None
# 5.2 创建一个路由处理函数：参数从post请求的请求体中提交
@app.post("/items/", summary="创建数据")
async def create_item(item: Item):
    dict = item.model_dump()
    print(dict)

    return {"item_name": item.name, "item_price": item.price, "item_is_offer": item.is_offer}


# 6. 常见的响应形式
# 6.1、路由处理函数返回一个 Pydantic 模型实例，FastAPI 将自动将其转换为 JSON 格式，并作为响应发送给客户端：
@app.post("/items/return", summary="返回 Pydantic 模型实例")
async def create_item(item: Item):

    #TODO 一些业务逻辑

    return item

# 6.2、使用 HTTPException 抛出异常，返回自定义的状态码和详细信息。
#以下实例在 item_id 为 42 会返回 404 状态码：
from fastapi import HTTPException

@app.delete("/items/{item_id}", summary="抛出异常")
async def read_item(item_id: int):

    # TODO 对管理员身份进行校验
    if item_id == 100:
        raise HTTPException(status_code=404, detail="商品找不到")

    # TODO 删除商品
    return {"item_id": item_id}

# 6.3、JSONResponse
@app.get("/api/user")
async def get_user():
    # 等价于直接 return {"name": "张三", "age": 20}（FastAPI 自动转 JSONResponse）
    return JSONResponse(
        content={"name": "张三", "age": 20},
        status_code=200,  # 可选，默认 200
        headers={"X-Custom-Header": "custom-value"}  # 可选，自定义响应头
    )

# 6.4、FileResponse
@app.get("/download/excel")
async def download_excel():
    excel_path = "D:/Agent_Learnings/LangGraph/test.xlsx"
    # 返回文件并指定下载文件名
    return FileResponse(
        path=excel_path,
        filename="月度报表.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# 6.5、PlainTextResponse
@app.get("/text")
async def get_text():
    return PlainTextResponse(content="<h1>这是纯文本响应</h1>", status_code=200)

# 6.6、HTMLResponse
@app.get("/hello")
async def hello(name: str = "游客"):
    html_content = f"""
    <html>
        <body>
            <h1>你好，{name}！</h1>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# 6.7、重定向(客户端跳转)
@app.get("/old-path")
async def redirect_old_path():

    # 重定向到 /new-path，状态码 307 表示临时重定向
    return RedirectResponse(url="/new-path", status_code=307)

@app.get("/new-path")
async def new_path():
    return {"message": "这是新接口"}


# 6.8、流式响应
async def generate_stream():
    # 模拟流式输出（逐字返回）
    words = ["你", "好", "，", "这", "是", "流", "式", "响", "应"]
    for word in words:
        await asyncio.sleep(0.5)
        yield word.encode("utf-8")  # 流式输出需返回字节流

@app.get("/stream")
async def stream_response():
    return StreamingResponse(generate_stream(), media_type="text/plain")

# 6.9、基础响应类
@app.get("/custom")
async def custom_response():
    # 返回二进制数据，指定自定义 MIME 类型
    return Response(
        content="<h1>html响应</h1>",
        # media_type="text/plain",
        media_type="text/plain",
        # media_type="application/json",
        status_code=200)

# 运行应用程序
# 命令行：--reload 热更新
# uv run uvicorn test.fastapi.01demo:app --reload

if __name__ == "__main__":
    # uvicorn.run("test.fastapi.01demo:app", host="127.0.0.1", port=8000, reload=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)