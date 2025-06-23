import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.set_abstraction import PointNet_SA_Module, PointNet_SA_Module_MSG


class pointnet2(nn.Module):
    def __init__(self, in_channels):
        super(pointnet2, self).__init__()

        self.pt_sa1 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=in_channels, mlp=[64, 64, 128], group_all=False)
        self.pt_sa2 = PointNet_SA_Module(M=128, radius=0.4, K=64, in_channels=133, mlp=[128, 128, 256], group_all=False)
        # self.pt_sa3 = PointNet_SA_Module(M=96, radius=0.8, K=128, in_channels=261, mlp=[256, 256, 512], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=261, mlp=[256, 512, 1024], group_all=True)

    def forward(self, points):
        output = []
        points = points.permute(0, 2, 1)
        xyz = points
        new_xyz, new_points = self.pt_sa1(xyz, points)
        output.append(new_points)
        new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
        output.append(new_points)
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)
        output.append(new_points)

        return output

class pointnet2_cls_ssg(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(pointnet2_cls_ssg, self).__init__()
        self.pt_sa1 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=in_channels, mlp=[64, 64, 128], group_all=False)
        self.pt_sa2 = PointNet_SA_Module(M=128, radius=0.4, K=64, in_channels=131, mlp=[128, 128, 256], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.cls = nn.Linear(256, nclasses)

    def forward(self, xyz, points):
        batchsize = xyz.shape[0]
        new_xyz, new_points = self.pt_sa1(xyz, points)
        new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)
        net = new_points.view(batchsize, -1)
        net = self.dropout1(F.relu(self.bn1(self.fc1(net))))
        net = self.dropout2(F.relu(self.bn2(self.fc2(net))))
        net = self.cls(net)
        return net


class pointnet2_cls_msg(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(pointnet2_cls_msg, self).__init__()
        self.pt_sa1 = PointNet_SA_Module_MSG(M=512, radiuses=[0.1, 0.2, 0.4], Ks=[16, 32, 128], in_channels=in_channels, mlps=[[32, 32, 64],
                                                   [64, 64, 128],
                                                   [64, 96, 128]])
        self.pt_sa2 = PointNet_SA_Module_MSG(M=128,
                                             radiuses=[0.2, 0.4, 0.8],
                                             Ks=[32, 64, 128],
                                             in_channels=323,
                                             mlps=[[64, 64, 128],
                                                   [128, 128, 256],
                                                   [128, 128, 256]])
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=643, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.cls = nn.Linear(256, nclasses)

    def forward(self, xyz, points):
        batchsize = xyz.shape[0]
        new_xyz, new_points = self.pt_sa1(xyz, points)
        new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)
        net = new_points.view(batchsize, -1)
        net = self.dropout1(F.relu(self.bn1(self.fc1(net))))
        net = self.dropout2(F.relu(self.bn2(self.fc2(net))))
        net = self.cls(net)
        return net


class cls_loss(nn.Module):
    def __init__(self):
        super(cls_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, pred, lable):
        '''

        :param pred: shape=(B, nclass)
        :param lable: shape=(B, )
        :return: loss
        '''
        loss = self.loss(pred, lable)
        return loss


if __name__ == '__main__':
    xyz = torch.randn(32, 5, 2048)
    points = torch.randn(32, 5, 2048)
    label = torch.randint(0, 40, size=(16, ))
    ssg_model = pointnet2(10)

    output = ssg_model(xyz)
    print(ssg_model)
    #net = ssg_model(xyz, points)
    #print(net.shape)
    #print(label.shape)
    #loss = cls_loss()
    #loss = loss(net, label)
    #print(loss)