'use strict';

function AliOrderDetailHelper(orderId)
{
    if (this === window) {
        return new AliOrderDetailHelper(orderId);
    }

    this.orderId = orderId;
    this.init();
}

AliOrderDetailHelper.prototype.init = function()
{
    this.initButton();
    this.initDialog();
    this.initEvents();
}

AliOrderDetailHelper.prototype.initButton = function()
{
    var button = BX.create('a', {
        props: {
            id: 'IPOL_ALI_ORDER_BUTTON',
            className: 'adm-btn',
            style: 'color: '+ this.getButtonColor(),
        },

        text: BX.message('IPOL_ALI_BUTTON_OPEN_DIALOG'),

        events: {
            click: BX.proxy(function(e) {
                this.dialog.Show();

                return BX.PreventDefault(e);
            }, this),
        },
    });

    var parent = document.querySelector('.adm-detail-toolbar .adm-detail-toolbar-right');

    if (parent.firstChild) {
        parent.insertBefore(button, parent.firstChild);
    } else {
        parent.appendChild(button);
    }
}

AliOrderDetailHelper.prototype.initDialog = function()
{
    this.dialog = new BX.CAdminDialog({
        title      : BX.message('IPOL_ALI_DIALOG_TITLE'),
        content_url: this.getFormUrl(),
        icon       : 'head-block',
        width      : 800,
        height     : 400,
        resizable  : true,
        draggable  : true,
        buttons    : [],
    });

    this.initDialogButtons();
}

AliOrderDetailHelper.prototype.initDialogButtons = function()
{

}

AliOrderDetailHelper.prototype.initEvents = function()
{
    var self = this;

    BX.addCustomEvent(this.dialog, 'onWindowRegister',function(){  
        this.GetForm().onsubmit = '';

        if (this.GetForm()['save']) {
            this.GetForm().removeChild(this.GetForm()['save']);
        }
    });

    $(document).on('submit', 'form[name="IPOL_ALI_ORDER"]', function(e) {
        self.process(e);

        return false;
    });
}

AliOrderDetailHelper.prototype.getFormUrl = function()
{
    return '/bitrix/admin/ipol.aliexpress_order_edit.php?ORDER_ID='+ this.orderId +'&bxpublic=Y';
}

AliOrderDetailHelper.prototype.process = function(action)
{
	if (this._ajaxRequest) {
		return;
    }

	var btn  = BX.type.isString(action) ? false : (action.originalEvent ? action.originalEvent.submitter : false);
    var form = this.dialog.GetForm();

    // return this.dialog.PostParameters('action=' + (
    //     BX.type.isString(action) 
    //         ? action 
    //         : btn.getAttribute('name')
    // ));

    this.showLoading(btn);
	this._ajaxRequest = BX.ajax({
		method : 'POST',
		url    : form.getAttribute('action') || document.location.href,
		data   : this.dialog.GetParameters()
			+ '&action=' + (
                    BX.type.isString(action) 
                        ? action 
                        : (btn ? btn.getAttribute('name') : '')
			)
		,
		dataType: 'html',
		onsuccess: BX.proxy(function(response) {
			try {
                this.dialog.ClearButtons();
                this.dialog.SetContent(response);

                // удаляем лишнее от битрикса
                this.dialog.GetForm().onsubmit = '';
                this.dialog.GetForm().removeChild(this.dialog.GetForm()['save']);

				// this.initEvents();
                // this.initDialogButtons();
                
				this.hideLoading(btn);
			} catch (e) {
				console.log(e);
			}

			this._ajaxRequest = false;
		}, this)
	});
}

AliOrderDetailHelper.prototype.showLoading = function(btn)
{
	this.hideLoading();
	
	this._loading = BX.showWait();
	this._loading_btn = btn ? this.dialog.showWait(btn) : false;
}

AliOrderDetailHelper.prototype.hideLoading = function(btn)
{
	this._loading && BX.closeWait(this._loading);
	this._loading_btn && this.dialog.closeWait(btn);

	this._loading = false;
	this._loading_btn = false;
}

AliOrderDetailHelper.prototype.getButtonColor = function()
{
    return '';
    
	var status = BX('IPOLH_DPD_ORDER_FORM').dataset.orderStatus;

	if (status == 'OrderError'
		|| status == 'Canceled'
	) {
		return '#cb4143';
	}

	if (status == 'OrderPending') {
		return '#3d7fb5';
	}

	if (status != 'NEW') {
		return 'green';
	}

	return '';
}