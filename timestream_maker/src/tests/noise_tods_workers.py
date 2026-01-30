
def worker_tod(curr_spec_list):
    global _args
    freq_fft, sample_freq, tod_len = _args 
    TOD_list = []
    for pk in curr_spec_list:
        TOD_list.append(gaussian_random_tod(freq_fft, pk, res = (1/sample_freq), nx = tod_len))
    return np.asarray(TOD_list)

def gaussian_tod_pll(freq_fft, curr_spec_list, sample_freq, tod_len, ncpus):
    with Pool(ncpus, initializer=worker_init, initargs=(freq_fft, sample_freq, tod_len )) as p:
        # Transform full cube (nchan, npix, npix) as (npix*npix, nchan)
        tod_list = p.map(worker_tod, np.array_split(curr_spec_list, ncpus) )
    tod_list_final = np.vstack(tod_list) 
    return tod_list_final

def worker_model(curr_sim_pspec_dic_for_worker):
    global _args
    freq_fft, sample_freq, tod_len = _args 
    spec_list = []
    for tod1, tod2 in curr_sim_pspec_dic_for_worker:
        spec_list.append(( np.fft.fft(tod1) * (1/sample_freq) * np.conj( np.fft.fft(tod2) * (1/sample_freq) ) / tod_len  ).real )
    return np.asarray(spec_list)
    curr_spec = ( np.fft.fft(tod1) * (1/sample_freq) * np.conj( np.fft.fft(tod2) * (1/sample_freq) ) / tod_len  ).real
    curr_sim_pspec_dic[(cntr1, cntr2)] = [freq_fft, curr_spec]

def model_pll(freq_fft, tod_sim_arr, sample_freq, tod_len, ncpus):

    curr_sim_pspec_dic_for_worker = []
    cntr = []
    for (cntr1, tod1) in enumerate( tod_sim_arr ):
        for (cntr2, tod2) in enumerate( tod_sim_arr ):
            if cntr2<cntr1: continue  
            else: 
                curr_sim_pspec_dic_for_worker.append((tod1, tod2))
                cntr.append((cntr1, cntr2))
    
    with Pool(ncpus, initializer=worker_init, initargs=(freq_fft, sample_freq, tod_len )) as p:
        curr_sim_pspec = p.map(worker_model, np.array_split(curr_sim_pspec_dic_for_worker, ncpus) )
    curr_sim_pspec = np.vstack(curr_sim_pspec) 

    curr_sim_pspec_dic = {}
    for curr_spec, (cntr1, cntr2) in zip(curr_sim_pspec, cntr):  
        curr_sim_pspec_dic[