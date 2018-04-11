

plt.clf()
fig = plt.gcf()
_ = fig.set_size_inches(15, 2 * 3.1)
gs = gridspec.GridSpec(2, 2, width_ratios=[6, 1], height_ratios=[1, 1],  hspace=0.0, wspace=0.03)

thres = 20
vmin = 50
vmax = 200

ax1 = plt.subplot(gs[0, :-1])

disp = plotting.plot_anat(str(fprep_std), display_mode='z',
                          cut_coords=[-15, -5, 10, 20, 40], cmap='cividis', threshold=thres, vmin=vmin, vmax=vmax,
                          axes=ax1)
disp.add_contours(str(ATLAS_HOME / '1mm_tpm_csf.nii.gz'), colors=['k'], levels=[0.8])
disp.add_contours(str(ATLAS_HOME / '1mm_tpm_wm.nii.gz'), colors=['w'], levels=[0.8], linewidths=[1], alpha=0.7)
disp.add_contours(str(ATLAS_HOME / '1mm_brainmask.nii.gz'), colors=['k'], levels=[0.8], linewidths=[3], alpha=.7)

ax2 = plt.subplot(gs[1, :-1])
disp = plotting.plot_anat(str(feat_std), display_mode='z',
                          cut_coords=[-15, -5, 10, 20, 40], cmap='cividis', threshold=thres, vmin=vmin, vmax=vmax,
                          axes=ax2)
disp.add_contours(str(ATLAS_HOME / '1mm_tpm_csf.nii.gz'), colors=['k'], levels=[0.8])
disp.add_contours(str(ATLAS_HOME / '1mm_tpm_wm.nii.gz'), colors=['w'], levels=[0.8], linewidths=[1], alpha=0.7)
disp.add_contours(str(ATLAS_HOME / '1mm_brainmask.nii.gz'), colors=['k'], levels=[0.8], linewidths=[3], alpha=.7)

ax1.annotate(
    'fMRIPrep',
    xy=(0., .5), xycoords='axes fraction', xytext=(-24, .0),
    textcoords='offset points', va='center', color='k', size=24,
    rotation=90)

ax2.annotate(
    r'\texttt{feat}',
    xy=(0., .5), xycoords='axes fraction', xytext=(-24, .0),
    textcoords='offset points', va='center', color='k', size=24,
    rotation=90)


inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1, 15],
                                              subplot_spec=gs[:, 1], wspace=0.01)

ax3 = fig.add_subplot(inner_grid[0])
gradient = np.hstack((np.zeros((50,)), np.linspace(0, 1, 120), np.ones((130,))))[::-1]
gradient = np.vstack((gradient, gradient))
ax3.imshow(gradient.T, aspect='auto', cmap=plt.get_cmap('cividis'))
ax3.xaxis.set_ticklabels([])
ax3.xaxis.set_ticks([])
ax3.yaxis.set_ticklabels([])
ax3.yaxis.set_ticks([])

ax4 = fig.add_subplot(inner_grid[1])
sn.distplot(nb.load(str(fprep_std)).get_data()[mask2mm], label='fMRIPrep',
            vertical=True, ax=ax4, kde=False, norm_hist=True)
sn.distplot(nb.load(str(feat_std)).get_data()[mask2mm], label=r'\texttt{feat}', vertical=True,
            color='darkorange', ax=ax4, kde=False, norm_hist=True)

# plt.gca().set_ylim((0, 300))
plt.legend(prop={'size': 20}, edgecolor='none')

ax4.xaxis.set_ticklabels([])
ax4.xaxis.set_ticks([])
ax4.yaxis.set_ticklabels([])
ax4.yaxis.set_ticks([])

plt.axis('off')
ax3.axis('off')
ax4.axis('off')

plt.savefig(str(out_folder / 'fmriprep-feat-std.pdf'),
            format='pdf', bbox_inches='tight', pad_inches=0.2, dpi=300)

