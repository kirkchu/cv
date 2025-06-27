import asyncio
import cv2
import numpy as np
from websockets.asyncio.server import serve


async def echo(websocket):
    async for message in websocket:
        # 將接收到的影像資料發送回去
        await websocket.send(message)


async def display(websocket):
    async for message in websocket:
        # 確保接收到的資料為 bytes 格式
        if isinstance(message, str):
            message = message.encode('utf-8')
        # 將接收到的影像資料轉換為 numpy 陣列
        img_data = np.frombuffer(message, dtype=np.uint8)
        # 解碼 JPEG 格式影像
        frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imshow("WebSocket Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


async def main():
    async with serve(display, "localhost", 8765) as server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
    cv2.destroyAllWindows()