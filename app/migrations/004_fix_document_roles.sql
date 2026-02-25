-- Clean up duplicate document records, keeping only the latest row per (tenant_id, filename)
DELETE FROM documents a USING documents b
WHERE a.ctid < b.ctid
  AND a.tenant_id = b.tenant_id
  AND a.filename = b.filename;

-- Ensure allowed_roles column defaults to 'admin'
ALTER TABLE documents ALTER COLUMN allowed_roles SET DEFAULT 'admin';

-- Any document where allowed_roles is NULL or empty should default to 'admin'
UPDATE documents SET allowed_roles = 'admin' WHERE allowed_roles IS NULL OR TRIM(allowed_roles) = '';
