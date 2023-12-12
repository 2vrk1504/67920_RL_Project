import os
import torch


def load_checkpoint(net, optimizer=None, step='max', save_dir='checkpoints'):
    # os.makedirs(save_dir, exist_ok=True)

    checkpoints = [x for x in os.listdir(save_dir) if not x.startswith('events')]
    if step == 'max':
        step = 0
        if checkpoints:
            step, last_checkpoint = max([(int(x.split('.')[0]), x) for x in checkpoints])
    else:
        last_checkpoint = str(step) + '.pth'
    if step:
        save_path = os.path.join(save_dir, last_checkpoint)
        state = torch.load(save_path, map_location='cpu')
        net.load_state_dict(state['net'])
        if optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Loaded checkpoint %s' % save_path)
    return step

def load_pv_checkpoint(net, step='max', save_dir='checkpoints'):
    # os.makedirs(save_dir, exist_ok=True)

    checkpoints = [x for x in os.listdir(save_dir) if not x.startswith('events')]
    if step == 'max':
        step = 0
        if checkpoints:
            step = max([int(x.split('_')[0]) for x in checkpoints])

    if step:
        last_p_checkpoint = str(step) + '_policy.pth'
        last_v_checkpoint = str(step) + '_value.pth'
        save_p_path = os.path.join(save_dir, last_p_checkpoint)
        save_v_path = os.path.join(save_dir, last_v_checkpoint)
        state = torch.load(save_p_path, map_location='cpu')
        net.policy_model.load_state_dict(state['net'])
        net.policy_optimizer.load_state_dict(state['optimizer'])

        state = torch.load(save_v_path, map_location='cpu')
        net.value_model.load_state_dict(state['net'])
        net.value_optimizer.load_state_dict(state['optimizer'])
        print('Loaded checkpoint %s' % save_p_path)
    return step


def load_Q_checkpoint(Qmodel, optimizer=None, step='max', save_dir='checkpoints'):
    # os.makedirs(save_dir, exist_ok=True)

    checkpoints = [x for x in os.listdir(save_dir) if not x.startswith('events')]
    if step == 'max':
        step = 0
        if checkpoints:
            step = max([int(x.split('_')[0]) for x in checkpoints])

    if step:
        for i in range(Qmodel.n):
            last_checkpoint = str(step) + '_action' + str(i) + '.pth'
            save_path = os.path.join(save_dir, last_checkpoint)
            state = torch.load(save_path, map_location='cpu')
            Qmodel.models[i].load_state_dict(state['net'])
            Qmodel.optimizers[i].load_state_dict(state['optimizer'])
    print('Loaded checkpoint %s' % save_dir + ' step: '+str(step))
    return step

def save_checkpoint(net, optimizer, step, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, str(step) + '.pth')

    torch.save(dict(net=net.state_dict(), optimizer=optimizer.state_dict()), save_path)
    print('Saved checkpoint %s' % save_path)

def save_pv_checkpoint(net, step, save_dir='checkpoints', log=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_p_path = os.path.join(save_dir, str(step) + '_policy.pth')
    save_v_path = os.path.join(save_dir, str(step) + '_value.pth')

    torch.save(dict(net=net.policy_model.state_dict(), optimizer=net.policy_optimizer.state_dict()), save_p_path)
    torch.save(dict(net=net.value_model.state_dict(), optimizer=net.value_optimizer.state_dict()), save_v_path)
    if log:
        print('Saved checkpoint %s' % save_p_path)


def save_Q_checkpoint(Qmodel, step, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(Qmodel.n):
        save_path = os.path.join(save_dir, str(step) +'_action'+str(i)+'.pth')
        torch.save(dict(net=Qmodel.models[i].state_dict(), optimizer=Qmodel.optimizers[i].state_dict()), save_path)
    print('Saved checkpoint %s' % save_dir + ' step: ' + str(step))

