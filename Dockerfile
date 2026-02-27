# KiroGate - Deno Edition
FROM denoland/deno:2.1.4

WORKDIR /app

# 复制依赖配置
COPY deno.json .

# 复制源码
COPY main.ts .
COPY lib/ ./lib/

# 创建数据目录
RUN mkdir -p /app/data && chown -R deno:deno /app

USER deno

# 缓存依赖
RUN deno cache main.ts

EXPOSE 8000

CMD ["deno", "run", "--allow-net", "--allow-env", "--allow-read", "--allow-write", "--unstable-kv", "main.ts"]
