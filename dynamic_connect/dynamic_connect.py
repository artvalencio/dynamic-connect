import cami
import mne
import numpy as np
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import gc
import matplotlib
matplotlib.use('Agg')


def dynamic_connect(eeg, window_length, overlap=0, metric='transfer_entropy',
                    min_freq=None, max_freq=None, threshold=None, video_title=None,
                    divs=np.arange(-0.5, 0.6, 0.2), tau=1, symbolic_length=1, delay=0,
                    events=None, reorder_chans_anim=None,framerate=10,font_scale=2):
    """dynamic_connect: builds a dynamic functional connectome from EEG data
    --------------------------
    Inputs:
    eeg: mne.Raw
        The pre-processed EEG time-series
    window_length: int
        The window length of analysis,  given in number of data points
    overlap: int,  optional
        The overlap between one interval and the next,  given in number
        of data points.
        Default: 0 (no overlap)
    metric: str,  optional
        The metric to be used for the construction of the connectome.
        Options:    - Transfer Entropy ('transfer_entropy')
                    - Mutual Information ('mutual_info')
                    - Pearson correlation ('pearson')
                    - Spearman correlation ('spearman')
        Default: 'transfer_entropy'
    min_freq: float,  None,  optional
        The minimum frequency for analysis (filter will be applied).
        Default: None (no filter is applied)
    max_freq: float,  None,  optional
        The minimum frequency for analysis (filter will be applied).
        Default: None (no filter is applied)
    threshold: float,  None,  optional
        The cutoff where the value obtained has significance.
        For example,  for metric='pearson' and threshold=0.75,  then
        only links with pearson correlation |r|>=0.75 are considered.
        Default: No (all links are considered)
    video_title: str,  None,  optional
        The title to the animation file with the dynamic functional
        connectome to be produced by this function.
        If None,  no video is produced.
        Default: None.
    divs: list,  tuple,  np.ndarray,  optional
        Partition divisions,  used for transfer_entropy or mutual_information
        metrics.
        If metrics='pearson' or 'spearman',  it is not used.
        Default: np.arange(-0.5, 0.6, 0.2)
        (equivalent to: [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])
    tau: int,  optional
        Time-delay of reconstruction,  used for transfer_entropy or
        mutual_information metrics. Usually taken as the first zero of
        autocorrelation function or the first minima of the (auto)mutual
        information.
        If metric='pearson' or 'spearman',  it is not used.
        Default: 1.
    symbolic_length: int,  list,  tuple,  optional
        The symbolic length to be used for transfer_entropy of
        mutual_information metrics.
        If metric='pearson' or 'spearman',  it is not used.
        Default: 1
    delay: int,  optional
        The time delay of transmission of information to be
        considered between two cortical areas. If a number different
        than zero is provided,  a lag is applied between the two
        time-series. Must be at least 10 times smaller than window_length.
        Default: 0
    events: ndarray, None, optional
        Array of MNE events, which can be the timings and types of stimulus
        presented, actions taken by the participant or the occurrence of
        clinically significant behaviours.
        Default: None. (no events array provided)
    reorder_chans_anim: list,  tuple,  None,  optional
        If the user wishes to reorder the channels for the animation,  so the
        adjacency matrix is clearer with respect to the brain positions, 
        then provide the list with the new channel order.
        If None,  it will not be reordered.
        This feature has no impact on the dFC np.ndarray output,  which will
        follow the original channel order of the eeg file.
        Default: None.
    framerate: int, optional
        The framerate for the animation generated.
        Default: 10
    font_scale: float, optional
        The font scale for the ticks in the animation generated.
        Default: 2.
    --------------------------
    Outputs:
    dfc: np.ndarray
        Array with the results,  of dimensions (time, n_chans, n_chans).
    The animation is saved to file of name given in video_title.
    If video_title is None,  no animation is produced.
    --------------------------
    Example:
    dfc=dynamic_connect.dynamic_connect(eeg, 1000, overlap=800, metric='transfer_entropy',
        min_freq=None, max_freq=None, threshold=0.5, video_title="DFC.mp4",
        divs=np.arange(-0.5, 0.6, 0.2), tau=1, symbolic_length=1, delay=0,events=None)
    """
    # checking condition for delay:
    if delay > window_length / 10:
        raise ValueError('delay must be at least 10 times smaller than window_length')
    # filtering
    if min_freq is None and max_freq is None:
        pass
    else:
        eeg = eeg.filter(l_freq=min_freq, h_freq=max_freq)
        # initialization
    ch_names = eeg.info['ch_names']
    x, times = eeg.get_data(return_times=True)
    chlen = len(x)
    total_time = x.shape[1] - window_length
    window_step = window_length - overlap
    w_times = range(0, total_time, window_step)
    ntimes = len(w_times)
    ntitles = [str(times[i]) + 's - ' + str(times[i + window_length]) + 's' for i in w_times]
    dfc = np.empty((ntimes, chlen, chlen))
    dfc.fill(np.nan)
    dfc2 = np.empty((ntimes, chlen, chlen)) #dfc2 will be the dfc without the elements below the threshold, for plotting only
    dfc2.fill(np.nan)

    # function to compute a single link
    def link(x, y, metric, divs, tau, symbolic_length, delay):
        if delay == 0:
            if metric == 'transfer_entropy':
                out = cami.transfer_entropy(x, y, x_divs=divs, y_divs=divs, tau=tau, symbolic_length=symbolic_length)
            elif metric == 'mutual_info':
                out = cami.mutual_info(x, y, x_divs=divs, y_divs=divs, tau=tau, symbolic_length=symbolic_length)
            elif metric == 'pearson':
                out = pearsonr(x, y)[0]
            elif metric == 'spearman':
                out = spearmanr(x, y).correlation
            else:
                out = np.nan()
        elif delay > 0:
            if metric == 'transfer_entropy':
                out = cami.transfer_entropy(x[:-delay], y[delay:], x_divs=divs, y_divs=divs, tau=tau,
                                            symbolic_length=symbolic_length)
            elif metric == 'mutual_info':
                out = cami.mutual_info(x[:-delay], y[delay:], x_divs=divs, y_divs=divs, tau=tau,
                                       symbolic_length=symbolic_length)
            elif metric == 'pearson':
                out = pearsonr(x[:-delay], y[delay:])[0]
            elif metric == 'spearman':
                out = spearmanr(x[:-delay], y[delay:]).correlation
            else:
                out = np.nan()
        else:  # delay<0
            if metric == 'transfer_entropy':
                out = cami.transfer_entropy(x[-delay:], y[:delay], x_divs=divs, y_divs=divs, tau=tau,
                                            symbolic_length=symbolic_length)
            elif metric == 'mutual_info':
                out = cami.mutual_info(x[-delay:], y[:delay], x_divs=divs, y_divs=divs, tau=tau,
                                       symbolic_length=symbolic_length)
            elif metric == 'pearson':
                out = pearsonr(x[-delay:], y[:delay])[0]
            elif metric == 'spearman':
                out = spearmanr(x[-delay:], y[:delay]).correlation
            else:
                out = np.nan()
        return out

    # build connectome
    wt=0
    print("calculating...")
    for t in range(0, total_time, window_step):
        event_added=False
        for ch1 in range(chlen):
            for ch2 in range(chlen):
                if ch1 != ch2:
                    dfc[wt, ch1, ch2] = link(x[ch1, t:t + window_length], x[ch2, t:t + window_length], metric=metric,
                                            divs=divs, tau=tau, symbolic_length=symbolic_length, delay=delay)
                    if threshold is not None:
                        if metric=='pearson' or metric=='spearman':
                            if abs(dfc[wt, ch1, ch2]) >= threshold:
                                dfc2[wt, ch1, ch2] = dfc[wt, ch1, ch2]
                            else:
                                dfc2[wt, ch1, ch2] = np.nan
                        else:
                            if dfc[wt, ch1, ch2] >= threshold:
                                dfc2[wt, ch1, ch2] = dfc[wt, ch1, ch2]
                            else:
                                dfc2[wt, ch1, ch2] = np.nan
                    else:
                        dfc2[wt,ch1,ch2] = dfc[wt,ch1,ch2]
                    if events is not None:
                        for eventidx in range(len(events)):
                            if (t<=events[eventidx,0]<=t+window_length) and (event_added is False):
                                ntitles[wt]=ntitles[wt]+", Event: "+str(events[eventidx,2])
                                event_added=True
                                break                                           
        wt+=1
    maxdfc=max(abs(np.nanmax(dfc)),abs(np.nanmin(dfc))) #used for coloraxis scaling

    # create animation
    if video_title != None:
        print("generating video... (may take a while)")
        print("(if an error occurs, please delete the temporary folder movie_frame \n before calling the function again)")
        #make frame temp directory
        os.mkdir("movie_frame")
        #build frames
        for t in range(wt):
            #set figure
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 11))
            # build dataframe
            df = pd.DataFrame(dfc2[t, :, :], columns=ch_names, index=ch_names)
            # reorder
            if reorder_chans_anim is not None:
                df=df.reindex(reorder_chans_anim).reindex(columns=reorder_chans_anim)
            # plot
            sns.set(font_scale=font_scale)
            if metric=='pearson' or metric=='spearman':
                axes = sns.heatmap(df, xticklabels=df.index, yticklabels=df.columns,mask=df.isnull(),
                            cmap='bwr', vmin=-maxdfc, vmax=maxdfc)
                if metric=='pearson':
                    axes.collections[0].colorbar.set_label("Pearson correlation",size=25)
                else:
                    axes.collections[0].colorbar.set_label("Spearman correlation",size=25)
            else:
                axes = sns.heatmap(df, xticklabels=df.index, yticklabels=df.columns,mask=df.isnull(),
                            cmap='Reds', vmin=0, vmax=maxdfc)
                if metric=='transfer_entropy':
                    axes.collections[0].colorbar.set_label("Transfer entropy",size=25)
                else:
                    axes.collections[0].colorbar.set_label("Mutual information",size=25)
            plt.title(ntitles[t],fontsize=30)
            plt.xlabel("Electrode location")
            plt.ylabel("Electrode location")
            plt.rc('axes', labelsize=25)
            fig.tight_layout()
            fig.savefig(f"movie_frame/frame_{t:05}.png", dpi=300)
            plt.cla()
            plt.clf()
            plt.close("all")
            plt.close(fig)
            gc.collect()
        #generate movie (requires ffmpeg)
        cmd="ffmpeg -framerate "+str(framerate)+" -i movie_frame/frame_%05d.png -pix_fmt yuv420p "+video_title
        os.system(cmd)
        #delete temporary folder with frames
        shutil.rmtree("movie_frame")

    # return results
    return dfc
