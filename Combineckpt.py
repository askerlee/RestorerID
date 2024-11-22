
import torch

def IPAsd_copy(IPAsd,crossattn_state_dict,midstr, num):
	IPAsd['model.diffusion_model.'+ midstr +'.to_q_lora.down.weight']= crossattn_state_dict[str(num) + '.to_q_lora.down.weight']
	IPAsd['model.diffusion_model.'+ midstr +'.to_q_lora.up.weight']= crossattn_state_dict[str(num) + '.to_q_lora.up.weight']
	IPAsd['model.diffusion_model.'+ midstr +'.to_k_lora.down.weight']= crossattn_state_dict[str(num) + '.to_k_lora.down.weight']
	IPAsd['model.diffusion_model.'+ midstr +'.to_k_lora.up.weight']= crossattn_state_dict[str(num) + '.to_k_lora.up.weight']
	IPAsd['model.diffusion_model.'+ midstr +'.to_v_lora.down.weight']= crossattn_state_dict[str(num) + '.to_v_lora.down.weight']
	IPAsd['model.diffusion_model.'+ midstr +'.to_v_lora.up.weight']= crossattn_state_dict[str(num) + '.to_v_lora.up.weight']
	IPAsd['model.diffusion_model.'+ midstr +'.to_out_lora.down.weight']= crossattn_state_dict[str(num) + '.to_out_lora.down.weight']
	IPAsd['model.diffusion_model.'+ midstr +'.to_out_lora.up.weight']= crossattn_state_dict[str(num) + '.to_out_lora.up.weight']
	if num//2*2 !=num:
		IPAsd['model.diffusion_model.'+ midstr +'.to_k_ip.weight']= crossattn_state_dict[str(num) + '.to_k_ip.weight']
		IPAsd['model.diffusion_model.'+ midstr +'.to_v_ip.weight']= crossattn_state_dict[str(num) + '.to_v_ip.weight']
	return IPAsd


def Combine(basemodel_ckpt_path,IDmodel_ckpt_path,save_path):
	basemodel_sd = torch.load(basemodel_ckpt_path, map_location="cpu")

	sd = basemodel_sd["state_dict"]
	
	ipa_sd = torch.load(IDmodel_ckpt_path, map_location="cpu")
	crossattn_state_dict = ipa_sd['ip_adapter']

	IPAsd = sd
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'input_blocks.1.1.transformer_blocks.0.attn1', 0)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'input_blocks.1.1.transformer_blocks.0.attn2', 1)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'input_blocks.2.1.transformer_blocks.0.attn1', 2)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'input_blocks.2.1.transformer_blocks.0.attn2', 3)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'input_blocks.4.1.transformer_blocks.0.attn1', 4)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'input_blocks.4.1.transformer_blocks.0.attn2', 5)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'input_blocks.5.1.transformer_blocks.0.attn1', 6)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'input_blocks.5.1.transformer_blocks.0.attn2', 7)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'input_blocks.7.1.transformer_blocks.0.attn1', 8)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'input_blocks.7.1.transformer_blocks.0.attn2', 9)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'input_blocks.8.1.transformer_blocks.0.attn1', 10)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'input_blocks.8.1.transformer_blocks.0.attn2', 11)

	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.3.1.transformer_blocks.0.attn1', 12)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.3.1.transformer_blocks.0.attn2', 13)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.4.1.transformer_blocks.0.attn1', 14)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.4.1.transformer_blocks.0.attn2', 15)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.5.1.transformer_blocks.0.attn1', 16)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.5.1.transformer_blocks.0.attn2', 17)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.6.1.transformer_blocks.0.attn1', 18)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.6.1.transformer_blocks.0.attn2', 19)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.7.1.transformer_blocks.0.attn1', 20)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.7.1.transformer_blocks.0.attn2', 21)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.8.1.transformer_blocks.0.attn1', 22)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.8.1.transformer_blocks.0.attn2', 23)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.9.1.transformer_blocks.0.attn1', 24)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.9.1.transformer_blocks.0.attn2', 25)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.10.1.transformer_blocks.0.attn1', 26)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.10.1.transformer_blocks.0.attn2', 27)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.11.1.transformer_blocks.0.attn1', 28)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'output_blocks.11.1.transformer_blocks.0.attn2', 29)

	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'middle_block.1.transformer_blocks.0.attn1', 30)
	IPAsd = IPAsd_copy(IPAsd,crossattn_state_dict, 'middle_block.1.transformer_blocks.0.attn2', 31)
	
	basemodel_sd["state_dict"] = IPAsd
	torch.save(basemodel_sd,save_path)


basemodel_ckpt_path = "ckpt/basemodel.ckpt"
IDmodel_ckpt_path = "ckpt/ip-adapter-faceid-plus_sd15.bin"
save_path = "ckpt/base+IDmodel.ckpt"
Combine(basemodel_ckpt_path,IDmodel_ckpt_path,save_path)