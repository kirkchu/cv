from deepface import DeepFace
import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "123456abcd",
    "database": "postgres",
}

IMGS = [
    "src/face/database/iu/image1.jpg",
    "src/face/database/tomcruise/image1.jpg",
    "src/face/database/eun-bin/image1.jpg",
]
NAMES = ["iu", "tomcruise", "eun-bin"]
SOURCE_IMG = "src/face/find_test2.jpg"
DIMENSION = 512


def vector_to_pg_string(values):
    return "[" + ",".join(f"{float(v):.10f}" for v in values) + "]"


def ensure_schema(conn, cur):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            embedding VECTOR(512) NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS face_embeddings_embedding_idx
        ON face_embeddings
        USING hnsw (embedding vector_l2_ops);
        """
    )
    cur.execute("TRUNCATE TABLE face_embeddings;")
    conn.commit()


def main():
    embeddings = []
    for img in IMGS:
        result = DeepFace.represent(img, model_name="Facenet512", detector_backend="retinaface")
        embeddings.append(result[0]["embedding"])

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    ensure_schema(conn, cur)

    for name, embedding in zip(NAMES, embeddings):
        cur.execute(
            "INSERT INTO face_embeddings (name, embedding) VALUES (%s, %s::vector)",
            (name, vector_to_pg_string(embedding)),
        )

    source_embedding = DeepFace.represent(
        SOURCE_IMG,
        model_name="Facenet512",
        detector_backend="retinaface",
    )[0]["embedding"]

    cur.execute(
        """
        SELECT name
        FROM face_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT 1;
        """,
        (vector_to_pg_string(source_embedding),),
    )
    matched_name = cur.fetchone()[0]

    conn.commit()
    cur.close()
    conn.close()

    print("索引庫準備完成！")
    print(f"索引庫大小: {len(NAMES)}")
    print(f"索引庫維度: {DIMENSION}")
    print(f"最相近結果: {matched_name}")

    try:
        import cv2

        img = cv2.imread(SOURCE_IMG)
        cv2.putText(img, matched_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as exc:
        print(f"顯示圖片時略過: {exc}")


if __name__ == "__main__":
    main()

