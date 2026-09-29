"""Strict read-only episode vector verification shared by census/postdeploy."""
CHECKER = r'''
def strict_episode_vectors(conn,db):
    import hashlib,re
    from hymem.dreaming.aggregate import load_clusterable_episodes
    if not db._load_vec_extension(conn):raise RuntimeError('vector_extension_unavailable')
    if not db.has_vec_table(conn,table='vec_episodes'):raise RuntimeError('episode_vector_table_missing')
    d=conn.execute("SELECT value FROM schema_meta WHERE key='vec_dim'").fetchone()
    m=conn.execute("SELECT value FROM schema_meta WHERE key='vec_model'").fetchone()
    if d is None or m is None:raise RuntimeError('vector_metadata_missing')
    dim=int(d[0]);model=m[0]
    if dim<=0 or not isinstance(model,str) or re.fullmatch(r'hymem-embedding-producer-v1:[0-9a-f]{64}',model) is None:
        raise RuntimeError('vector_metadata_invalid')
    expected={};unverifiable=0
    for episode in load_clusterable_episodes(conn,max_rowid=None,embedding_model=model,embedding_dim=dim):
        vec=db._finite_vec(episode['vector'],dim)
        if vec is None:
            unverifiable+=1
            continue
        key=int(episode['rowid'])
        if key in expected:raise RuntimeError('duplicate_authoritative_vector_key')
        expected[key]=db._pack_vector(vec)
    actual={}
    for row in conn.execute('SELECT rowid,embedding FROM vec_episodes'):
        key=int(row[0])
        if key in actual:raise RuntimeError('duplicate_actual_vector_key')
        actual[key]=bytes(row[1])
    missing=set(expected)-set(actual);surplus=set(actual)-set(expected)
    different={key for key in set(expected)&set(actual) if expected[key]!=actual[key]}
    def fingerprint(items):
        h=hashlib.sha256()
        for key,value in sorted(items.items()):
            h.update(str(key).encode()+b':'+str(len(value)).encode()+b':'+value)
        return h.hexdigest()
    return {'available':True,'verifiable':unverifiable==0,'aligned':not(missing or surplus or different or unverifiable),
            'unverifiable_episodes':unverifiable,
            'expected_count':len(expected),'actual_count':len(actual),'surplus_count':len(surplus),
            'missing_count':len(missing),'different_count':len(different),
            'expected_sha256':fingerprint(expected),'actual_sha256':fingerprint(actual)}
'''
