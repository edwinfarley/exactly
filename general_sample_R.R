library('reticulate')
permute_sample_general = function(family, dataframe, block, formula, N, I, t){
	n = nrow(dataframe)
	print(n)
	pandas = import('pandas')
	source_python('master.py')
	#source_python('perm_sample_norm_r.py')
	py_run_file('master.py')
	#py_run_file('perm_sample_norm_r.py')
	
	df = pandas$DataFrame$from_dict(r_to_py(dataframe))
	B = r_to_py(as.integer(block))
	f = r_to_py(formula)
	N = r_to_py(as.integer(N))
	I = r_to_py(as.integer(I))
	t = r_to_py(as.integer(t))
	print(f, N, I, t)
	if(family == 'Normal'){
		res = permute_search_normal(df, B, f, N, I, t)}
	else if(family == 'Binomial'){
		res = permute_search_logistic(df, B, f, N, I, t)}
	else if(family == 'Poisson'){
		res = permute_search_pois(df, B, f, N, I, t)}
	return(list(as.numeric(res[[1]]), res[[2]]))
}