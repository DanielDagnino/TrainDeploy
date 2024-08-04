import h5py


def join_h5(fn_assets_queries, query_feats, query_id, reference_feats, reference_id):
    with h5py.File(fn_assets_queries, 'w') as f_out:
        f_out.create_dataset('query', data=query_feats)
        f_out.create_dataset('query_ids', data=query_id)
        f_out.create_dataset('reference', data=reference_feats)
        f_out.create_dataset('reference_ids', data=reference_id)
