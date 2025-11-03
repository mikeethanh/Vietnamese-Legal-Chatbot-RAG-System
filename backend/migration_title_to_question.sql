-- Migration script to rename column title to question in document table
-- Run this script in your MariaDB database

USE demo_bot;

-- Step 1: Rename column and update datatype
ALTER TABLE document
CHANGE COLUMN title question VARCHAR(2000) NOT NULL DEFAULT '';

-- Verify the change
DESCRIBE document;

-- Optional: Check existing data
SELECT COUNT(*) as total_documents FROM document;
SELECT id, question, LEFT(content, 50) as content_preview FROM document LIMIT 5;
