# Example chatbot project

## Deploy
    docker compose up -d --build

## Check log
    docker logs -f chatbot-api
    docker logs -f chatbot-worker

## Database
    docker exec -it mariadb-tiny bash
    mysql -u root -p

    CREATE DATABASE demo_bot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
    
    USE demo_bot;

    CREATE TABLE chat_conversations (
        id INT NOT NULL AUTO_INCREMENT,
        conversation_id VARCHAR(50) NOT NULL DEFAULT '',
        bot_id VARCHAR(100) NOT NULL,
        user_id VARCHAR(100) NOT NULL,
        message TEXT,
        is_request BOOLEAN DEFAULT TRUE,
        completed BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id)
    );

    CREATE TABLE document (
        id INT NOT NULL AUTO_INCREMENT,
        title VARCHAR(100) NOT NULL,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id)
    );

### Vector DB
    http://localhost:6333/dashboard#/collections/llm


## References
- https://fastapi.tiangolo.com/tutorial/first-steps/
- https://derlin.github.io/introduction-to-fastapi-and-celery/03-celery/
- https://testdriven.io/courses/fastapi-celery/getting-started/