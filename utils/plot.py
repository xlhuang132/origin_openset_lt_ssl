import matplotlib.pyplot as plt
import numpy as np
from utils.color_generator import generate_colors,generate_ood_color
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE  
import torch
import os
import seaborn as sns
from sklearn import manifold
from scipy.interpolate import interp1d
import mpl_toolkits.axisartist as axisartist

def plot_group_acc_over_epoch(group_acc,legend=['Many','Medium','Few'],title="Group Acc",step=1,save_path='',warmup_epoch=0):
    assert len(group_acc)>0 and len(group_acc[0])==len(legend)
    group_acc=np.array(group_acc)
    many=group_acc[:,0]
    medium=group_acc[:,1]
    few=group_acc[:,2]
    x=[i*step for i in range(1,len(group_acc)+1)]
 
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(x, many, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.plot(x, medium, marker='o', markersize=3)
    plt.plot(x, few, marker='o', markersize=3) 
    # if warmup_epoch!=0:
    #     plt.axvline(2, color='r')
    plt.legend(legend)
    if save_path!='':
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def plot_loss_over_epoch(losses,title,save_path=''):
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    x=[i for i in range(1,len(losses)+1)]
    plt.plot(x, losses, markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    if save_path!='':
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_acc_over_epoch(acc,title="Average Accuracy",save_path=''):
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    x=[i for i in range(1,len(acc)+1)]
    plt.plot(x, acc, markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    if save_path!='':
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

  
def plot_ood_dectector_FPR95(results,xlegend=[],save_path="",xlabel="thresh",ylabel='FPR95'):
    """
        inputs:
            三维：[IF,thresh,TPF/FPR]
        第三维用不同形状的线代替 
    """
    
    return 


def plot_ood_dectector_TPR(results,line_names,x=[],title='',save_path="",xlabel="thresh",ylabel='TPR'):
    """
        inputs:
            三维：[IF,thresh,TPF/FPR]
        第三维用不同形状的线代替 
    """
    assert len(results)==len(name)
    plt.title('{}'.format(title))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if x and x[0] is not str:
        plt.xticks(x)
    markers=['o','x','*','^','+']
    linestyles=[':','--','-',"''",]
    for i,item in enumerate(results):  # if 
        for j,it in enumerate(item):
            plt.plot(x, it, marker=markers[i],linestyle=':', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.legend(line_names)
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    return 

def plot_feat_tsne(feat,y,title,num_classes,save_path=""):
    # 生成颜色
    fontsize=20
    colors = generate_colors(num_classes) 
    c_p = []
    if torch.is_tensor(y): yy = y.numpy()
    else: yy = y
    if -1 in yy:
        colors[-1]=generate_ood_color()
    for i in yy: 
        c_p.append(colors[i])
    X_tsne = TSNE(n_components=2,random_state=33,early_exaggeration=30).fit_transform(feat.numpy()) 
    # mix
    plt.figure(figsize=(12,12)) 
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=10, c=c_p,alpha=0.5)
    
    labels=[mpatches.Patch(color=colors[i],label=i)for i in range(len(colors))]
    plt.legend(handles=labels,loc='upper right',fontsize=fontsize) 
    # plt.title(title,fontsize=fontsize)    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="":
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  
    plt.close()
    return  


def plot_dl_du_all_feat_tsne(feat,y,title,num_classes,save_path="",dl_len=None,du_len=None):
    # 生成颜色
    fontsize=20
    if dl_len==None and du_len==None:
        dl_len=feat.size(0)
    colors = generate_colors(num_classes) 
    c_p = []
    if torch.is_tensor(y): yy = y.numpy()
    else: yy = y
    if -1 in yy:
        colors.append(generate_ood_color())
    for i in yy: 
        c_p.append(colors[i])
    X_tsne = TSNE(n_components=2,random_state=33,early_exaggeration=30).fit_transform(feat.numpy()) 
    # mix
    plt.figure(figsize=(12,12)) 
    
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=10, c=c_p[:],alpha=0.5)
    # plt.scatter(X_tsne[:dl_len, 0], X_tsne[:dl_len, 1],s=10, c=c_p[:dl_len],alpha=0.5,marker='o')
    # plt.scatter(X_tsne[dl_len:, 0], X_tsne[dl_len:, 1],s=10, c=c_p[dl_len:],alpha=0.5,marker='*')
    
    labels=[mpatches.Patch(color=colors[i],label=i)for i in range(len(colors))]
    plt.legend(handles=labels,loc='upper right',fontsize=fontsize) 
    plt.title(title,fontsize=fontsize)    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="":
        plt.savefig(os.path.join(save_path,"allfeat_tsne_best.jpg"), dpi=300, bbox_inches='tight')  
    plt.close()
    
    # dl
    plt.figure(figsize=(12,12)) 
    plt.scatter(X_tsne[:dl_len, 0], X_tsne[:dl_len, 1],s=10, c=c_p[:dl_len],alpha=0.5,marker='o') 
    
    labels=[mpatches.Patch(color=colors[i],label=i)for i in range(len(colors))]
    plt.legend(handles=labels,loc='upper right',fontsize=fontsize) 
    plt.title(title,fontsize=fontsize)    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="":
        plt.savefig(os.path.join(save_path,"allfeat_dl_tsne_best.jpg"), dpi=300, bbox_inches='tight')  
    plt.close()
    
    # du
    plt.figure(figsize=(12,12))  
    plt.scatter(X_tsne[dl_len:dl_len+du_len, 0], X_tsne[dl_len:dl_len+du_len, 1],s=10, c=c_p[dl_len:dl_len+du_len],alpha=0.5,marker='*')
    
    labels=[mpatches.Patch(color=colors[i],label=i)for i in range(len(colors))]
    plt.legend(handles=labels,loc='upper right',fontsize=fontsize) 
    plt.title(title,fontsize=fontsize)    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="":
        plt.savefig(os.path.join(save_path,"allfeat_du_tsne_best.jpg"), dpi=300, bbox_inches='tight')  
    plt.close()
     # du
    plt.figure(figsize=(12,12))  
    plt.scatter(X_tsne[dl_len+du_len:, 0], X_tsne[dl_len+du_len:, 1],s=10, c=c_p[dl_len+du_len:],alpha=0.5,marker='*')
    
    labels=[mpatches.Patch(color=colors[i],label=i)for i in range(len(colors))]
    plt.legend(handles=labels,loc='upper right',fontsize=fontsize) 
    plt.title(title,fontsize=fontsize)    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="":
        plt.savefig(os.path.join(save_path,"allfeat_test_tsne_best.jpg"), dpi=300, bbox_inches='tight')  
    plt.close()
    return  


def plot_ablation_feat_tsne(gt,pred,feat,num_classes=0,title='Test data feature',alg='',
                filenames=[],save_path=''):
    assert filenames!=[] and len(filenames)==4
    
    fontsize=20
    colors = generate_colors(num_classes,return_deep_group=True) 
    colors_r,colors_w=colors[:num_classes],colors[num_classes:]
    c_p = []
    if torch.is_tensor(gt): gt = gt.numpy()
    else: gt = gt
    n=len(gt)//len(filenames)
    for i in range(len(gt)): 
        c_p.append(colors_r[gt[i]]) if gt[i]==pred[i] else c_p.append(colors_w[gt[i]]) 
    X_tsne = TSNE(n_components=2,random_state=33,early_exaggeration=30).fit_transform(feat.numpy()) 
    
    # mix
    # plt.figure(figsize=(12,12))  
    fig = plt.figure(figsize=(16, 16))
    # 221代表创建2行2列一共4个子图，并从左往右第1个子图开始绘图。
    ax1 = fig.add_subplot(221)  
    ax1.set_title(filenames[0], y=-0.15, fontsize=fontsize) 
    ax1.scatter(X_tsne[:n, 0], X_tsne[:n, 1],s=10, c=c_p[:n],alpha=0.5) 

    ax2 = fig.add_subplot(222) 
    ax2.scatter(X_tsne[n:2*n, 0], X_tsne[n:2*n, 1],s=10, c=c_p[n:2*n],alpha=0.5) 
    ax2.set_title(filenames[1], y=-0.15, fontsize=fontsize) 

    ax3 = fig.add_subplot(223)  
    ax3.scatter(X_tsne[2*n:3*n, 0], X_tsne[2*n:3*n, 1],s=10, c=c_p[2*n:3*n],alpha=0.5) 
    ax3.set_title(filenames[2], y=-0.15, fontsize=fontsize) 

    ax4 = fig.add_subplot(224)  
    ax4.scatter(X_tsne[3*n:, 0], X_tsne[3*n:, 1],s=10, c=c_p[3*n:],alpha=0.5) 
    ax4.set_title(filenames[3], y=-0.15, fontsize=fontsize) 

    labels=[mpatches.Patch(color=colors_r[i],label="Correct-{}".format(i))for i in range(len(colors_r))]+\
    [mpatches.Patch(color=colors_w[i],label="Wrong-{}".format(i))for i in range(len(colors_w))]
    # plt.legend(handles=labels,loc='upper left',fontsize=10)  
    plt.legend(handles=labels,loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)  ##设置ax4中legend的位置，将其放在图外
 
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="":
        filename="test_data_ablation_feat_tsne.jpg" 
        plt.savefig(os.path.join(save_path,filename), dpi=300, bbox_inches='tight')  
    plt.close()
    
    return


def plot_problem_feat_tsne(gt,pred,feat,num_classes=0,title='Test data feature',alg='',
                filename='',save_path=''):
    fontsize=20
    colors = generate_colors(num_classes,return_deep_group=True) 
    colors_r,colors_w=colors[:num_classes],colors[num_classes:]
    c_p = []
    if torch.is_tensor(gt): gt = gt.numpy()
    else: gt = gt
    
    for i in range(len(gt)): 
        c_p.append(colors_r[gt[i]]) if gt[i]==pred[i] else c_p.append(colors_w[gt[i]]) 
    X_tsne = TSNE(n_components=2,random_state=33,early_exaggeration=30).fit_transform(feat.numpy()) 
    # mix
    plt.figure(figsize=(12,12)) 
    for i in range(len(X_tsne)):
        if gt[i]==pred[i]:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1],s=10, c=c_p[i],marker='o',alpha=0.5) 
        else:            
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1],s=10, c=c_p[i],marker='+',alpha=0.5) 
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=10, c=c_p[:],alpha=0.5) 
    labels=[mpatches.Patch(color=colors_r[i],label="Correct-{}".format(i))for i in range(len(colors_r))]+\
    [mpatches.Patch(color=colors_w[i],label="Wrong-{}".format(i))for i in range(len(colors_w))]
    plt.legend(handles=labels,loc='upper left',fontsize=10) 
    # plt.title(title,fontsize=fontsize)    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="":
        filename="test_data_problem_tsne_{}.jpg".format(alg) if filename=='' else filename
        plt.savefig(os.path.join(save_path,filename), dpi=300, bbox_inches='tight')  
    plt.close()
    
    return
def plot_accs_zhexian(group_accs,branches_name,title,x_legend,save_path="",xlabel="",ylabel='Accuracy',color=[]):
    assert len(branches_name)==len(group_accs)
    assert (len(branches_name)==len(color))if color!=[] else True
    plt.title('{}'.format(title))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if x_legend.any() is not str:
        plt.xticks(x_legend)
    # if color==[]:
    #     color=generate_colors(len(branches_name))
    # for i,group_acc in enumerate(group_accs):
    #     plt.plot(x_legend, group_acc, marker='o', markersize=3,c=color[i])  # 绘制折线图，添加数据点，设置点的大小
     
    for i,group_acc in enumerate(group_accs):
        plt.plot(x_legend, group_acc, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    
    plt.legend(branches_name)
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    
def plot_feat_cha_zhexian(gt_mean_cha,gt_var_cha,pred_mean_cha,pred_var_cha,num_classes=10,alg='',
                    filename='',save_path=''):
    fontsize=12
    fig = plt.figure(figsize=(6, 4))  
    x=range(num_classes)
    plt.xticks(x)
    plt.plot(x, gt_mean_cha, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.plot(x, gt_var_cha, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    # plt.plot(x, pred_mean_cha, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    # plt.plot(x, pred_var_cha, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    # plt.legend(['gt_mean_diff','gt_var_diff','pred_mean_diff','pred_var_diff'],fontsize=fontsize)
    plt.legend([r"$\Vert {\mu}^{train}_c-\mu^{test}_c \Vert$",r"$\Vert \sigma^{train}_c-\sigma^{test}_c \Vert$"],fontsize=fontsize)
    
    if save_path!="":
        filename="test_data_diff_{}.jpg".format(alg) if filename=='' else filename
        plt.savefig(os.path.join(save_path,filename), dpi=300, bbox_inches='tight')
    plt.close() 
    
    return  
def plot_feat_diff(means,algs,num_classes=10,save_path=''):
    fontsize=12
    fig = plt.figure(figsize=(6, 4))  
    x=range(num_classes)
    plt.xticks(x,fontsize=fontsize)
    for i in range(len(means)):
        plt.plot(x, means[i], marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.legend(algs,fontsize=fontsize)
    plt.fill_between([0, 2.5],0,0.4, facecolor='#ff7f0e', alpha=0.3) 
    plt.fill_between([2.5, 5.5],0,0.4, facecolor='#1f77b4', alpha=0.3) 
    plt.fill_between([5.5, 9],0,0.4, facecolor='#ee82ee', alpha=0.3) 
    
    if save_path!="":
        plt.savefig(os.path.join(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    return
   
def plot_task_bar(labeled_dist,unlabeled_dist,save_path):
    x_legend_=[i for i in range(len(labeled_dist))]
    cubic_interploation_model=interp1d(x_legend_,labeled_dist,kind="cubic")
    x_legend=np.linspace(x_legend_[0],x_legend_[-1],500)
    labeled_y=cubic_interploation_model(x_legend)
    
    
    cubic_interploation_model=interp1d(x_legend_,unlabeled_dist,kind="cubic")
    unlabeled_y=cubic_interploation_model(x_legend)
    
    fig = plt.figure(figsize=(6, 4))
    # myfonts = "Times New Roman"    
    # plt.rcParams['font.family'] = "sans-serif"
    # plt.rcParams['font.sans-serif'] = myfonts
    # plt.rc('font',family='Times New Roman')
    #使用axisartist.Subplot方法创建一个绘图区对象ax
    ax = axisartist.Subplot(fig, 111)  
    #将绘图区对象添加到画布中
    fig.add_axes(ax)
    
    ax.axis[:].set_visible(False)#通过set_visible方法设置绘图区所有坐标轴隐藏
    ax.axis["x"] = ax.new_floating_axis(0,0)#ax.new_floating_axis代表添加新的坐标轴
    ax.axis["x"].set_axisline_style("->", size = 1.0)#给x坐标轴加上箭头
    #添加y坐标轴，且加上箭头
    ax.axis["y"] = ax.new_floating_axis(1,0)
    ax.axis["y"].set_axisline_style("-|>", size = 1.0)
    #设置x、y轴上刻度显示方向
    ax.axis["x"].set_axis_direction("top")
    ax.axis["y"].set_axis_direction("right") 
    #给x坐标轴加上箭头
    ax.axis["x"].set_axisline_style("->", size = 1.0)
    ax.axis["y"].set_axisline_style("->", size = 1.0)
    ax.axis["x"].label.set_visible(True)
    ax.axis["x"].label.set_text("Class ID")
    ax.axis["x"].label.set_pad(-20)
    ax.axis["y"].label.set_visible(True)
    ax.axis["y"].label.set_text("Number")
    ax.axis["y"].label.set_pad(-20)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    plt.xticks([]) 
    plt.yticks([]) 
    # plt.xlabel("Class ID")
    # plt.ylabel("Number")
    y0=np.zeros_like(labeled_y)     
    plt.fill_between(x_legend, unlabeled_y, y0, where=(unlabeled_y > y0),facecolor='#1f77b4',alpha=0.3,label='unlabeled ID data') 
    plt.fill_between(x_legend, labeled_y, y0, where=(labeled_y > y0),facecolor='#1f77b4',label='labeled ID data')
    x_ood=np.array([i+len(unlabeled_dist)-1 for i in range(5)])
    y_ood=np.array([max(unlabeled_y)]*5)
    y2=np.zeros_like(x_ood) 
    plt.fill_between(x_ood, y_ood, y2, where=(y_ood > y2), facecolor='#ff7f0e', alpha=0.3,label='unlabeled OOD data') 
    plt.legend()
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    return 
   
def plot_task_stack_bar(labeled_dist,unlabeled_dist,save_path,title=""):
    x_legend_=[i for i in range(len(labeled_dist))] 
    
    fig = plt.figure(figsize=(6, 4))  
    
    plt.xticks([]) 
    plt.yticks([]) 
    plt.xlabel("Class ID")
    plt.ylabel("Number")
    plt.title(title)
    y0=np.zeros_like(labeled_dist)     
    plt.bar(x_legend_, labeled_dist, color='#1f77b4',label='labeled ID data')
    plt.bar(x_legend_, unlabeled_dist,  bottom=labeled_dist,color='#1f77b4',alpha=0.3,label='unlabeled ID data') 
    x_ood=np.array([i+len(unlabeled_dist) for i in range(5)])
    y_ood=np.array([max(unlabeled_dist)]*5)
    y2=np.zeros_like(x_ood)  
    plt.fill_between(x_ood, y_ood, y2, where=(y_ood > y2), facecolor='#ff7f0e', alpha=0.3,label='unlabeled OOD data') 
    # plt.bar(x_ood, y_ood, color='#ff7f0e', alpha=0.3,label='unlabeled OOD data') 
    plt.legend()
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    return 

def plot_ood_detect_feat(unlabeled_feat,unlabeled_y,pred_id,pred_ood):
    # 分对的ID用绿色三角形表示，分对的OOD用绿色圆形表示
    # 分错的ID用不同颜色的×形表示，分错的OOD用红色的圆形表示
    # 都为0的用灰色表示，表示没用到这部分数据
    no_detect=(1-pred_id)*(1-pred_ood)
    no_detect_index=torch.nonzero(no_detect,as_tuple=False).squeeze(1)
    ones=torch.ones_like(unlabeled_y)
    zeros=torch.zeros_like(unlabeled_y)
    gt_id =torch.where(unlabeled_y >= 0,ones,zeros)
    id_correct_index= torch.nonzero(gt_id==pred_id,as_tuple=False).squeeze(1)
    ood_correct_index= torch.nonzero((1-gt_id)==pred_ood,as_tuple=False).squeeze(1)
    id_wrong_index= torch.nonzero(gt_id==pred_ood,as_tuple=False).squeeze(1)
    ood_wrong_index= torch.nonzero((1-gt_id)==pred_id,as_tuple=False).squeeze(1)
    assert no_detect.sum()+id_correct_index.sum+ood_correct_index.sum()+id_wrong_index.sum()+ood_wrong_index.sum()==unlabeled_y.size(0)
    
    return 
def plot_bar(y,save_path=None):
    x=[i for i in range(len(y))]
    plt.bar(x, y)
    plt.xticks([]) 
    plt.yticks([]) 
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    return 


def plot_multi_bars(datas,labels, legend=None,xlabel="",ylabel="",title="",save_path="", tick_step=1, group_gap=0.2, bar_gap=0):
    '''
    legend : x轴坐标标签序列
    datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
    tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
    group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
    bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    ''' 
    # x为每组柱子x轴的基准位置
    x = np.arange(len(labels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # 绘制柱子
    for index, y in enumerate(datas):
        plt.bar(x + index*bar_span, y, bar_width) 
    plt.title(title)
    # ticks为新x轴刻度标签位置，即每组柱子x轴上的中心位置
    ticks = x + (group_width - bar_span) / 2
    plt.xticks(ticks, labels) 
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    return 